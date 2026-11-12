""" Write and read RTP audio (L24, as used by AES67) inside a .pcap capture file.

Everything is done with the standard library 'struct' module so you can see
every header byte, much like parsing a packet in C.
"""

import struct

import numpy as np

PCAP_GLOBAL = struct.Struct("<IHHiIII") #magic, ver major, tz, sigfigs, snaplen, linktype
PCAP_RECORD = struct.Struct("<IIII")    #ts_sec, ts_usec, incl_len, orig_len
RTP_HEADER = struct.Struct("!BBHII")    #V/P/X/CC, M/PT, seq, timestamp, SSRC ("!" = network byte order)

#------ L24 sample packing ------

def float_to_l24(samples):
    """Float -1..+1 -> big-endian 24-bit signed integer bytes."""
    ints = np.clip(np.round(samples*(2 ** 23 -1)), -2 ** 23 - 1, 2 ** 23 - 1).astype(np.int32)
    b = ints.astype(">i4").tobytes()
    return b"".join(b[i + 1:i +4] for i in range(0, len(b), 4))     #drop the top byte

def l24_to_float(payload):
    """Big-endian 24-bit signed bytes -> float -1..+1."""
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    ints = (raw[:,0] << 16) | (raw[:,1] << 8) | raw[:,2]
    ints = np.where(ints & 0x800000, ints - (1<<24), ints)  #sign-extended
    return ints / (2 ** 23 - 1)

# ------ header builders ------

def _ip_checksum(header):
    total = sum(struct.unpack("!10H", header))
    while total > 0xFFFF:
        total = (total&0xFFFF)+ (total >>16)
    return ~total & 0xFFFF

def _udp_ipv4_eth(payload, src_ip, dst_ip, src_port, dst_port):
    udp = struct.pack("!HHHH", src_port, dst_port, 8 + len(payload), 0) + payload
    ip  = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0, 
                      bytes(map(int, src_ip.split("."))), bytes(map(int, dst_ip.split("."))))
    ip = ip[:10] + struct.pack("!H", _ip_checksum(ip)) +ip[12:]
    eth = bytes.fromhex("01005e000001") + bytes.fromhex("020000000001") + struct.pack("!H", 0x0800)
    return eth + ip + udp

def write_rtp_pcap(path, audio, fs=48000, samples_per_packet=48, payload_type=97,
                   drop_packets=(), dst_ip="239.69.0.1", dst_port=5004):
    """ Packetisse mono float audio into RTP/L24 and save a .pcap file.
    
    drop_packets: packet indices to leave out, to simulate network loss.
    """
    drop = set(drop_packets)
    with open(path, "wb") as f:
        f.write(PCAP_GLOBAL.pack(0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)) #linktype 1 = Ethernet
        ssrc = 0x12345678
        n_packets = len(audio) // samples_per_packet
        for i in range(n_packets):
            if i in drop:
                continue
            chunk = audio[i * samples_per_packet:(i+1) * samples_per_packet]
            rtp = RTP_HEADER.pack(0x80, payload_type, i & 0xFFFF,
                                  (i * samples_per_packet) & 0xFFFFFFFF, ssrc)
            frame = _udp_ipv4_eth(rtp + float_to_l24(chunk), "192.168.1.10", dst_ip, 5004, dst_port)
            t = i * samples_per_packet / fs
            f.write(PCAP_RECORD.pack(int(t), int((t%1) *1e6), len(frame), len(frame)))
            f.write(frame)
        return n_packets

def read_rtp_pcap(path, dst_port=5004):
        """ Read a .pcap of RTP/L24 audio.  Returns (audio, stats).
        
        Lost packets are filled with silence so the timeline stays correct.
        """
        with open(path, "rb") as f:
            data = f.read()
        magic = struct.unpack("<I", data[:4])[0]
        if magic != 0xA1B2C3D4:
            raise ValueError("Not a classic little-endian microsecond .pcap file (pcapng is not supported)")
        off = PCAP_GLOBAL.size
        packets = {}
        while off + PCAP_RECORD.size <= len(data):
            _, _, incl_len, _ = PCAP_RECORD.unpack_from(data, off)
            off += PCAP_RECORD.size
            frame = data[off:off + incl_len]
            off += incl_len
            if struct.unpack("!H", frame[12:14])[0] != 0x0800:  #not IPv4
                continue
            ih1 = (frame[14] & 0x0F) *4
            if frame[23] != 17:  #not UDP
                continue
            udp = 14 + ih1
            if struct.unpack("!H", frame[udp + 2:udp + 4]) [0] != dst_port:
                continue
            rtp = udp + 8
            vpxcc, mpt, seq, ts, ssrc = RTP_HEADER.unpack_from(frame, rtp)
            header_len = 12 + 4 * (vpxcc & 0x0F)  # skip any CSRC entries
            packets[seq] = (ts, frame[rtp + header_len:])

        seqs = sorted(packets)
        spp = len(packets[seqs[0]][1]) // 3
        expected = seqs[-1] - seqs[0] + 1
        audio = np.zeros(expected * spp)
        for s in seqs:
            i = s - seqs[0]
            audio[i * spp:(i + 1) * spp] = l24_to_float(packets[s][1])
        lost = sorted(set(range(seqs[0], seqs[-1] + 1)) - set(seqs))
        stats = {"received": len(seqs), "expected": expected, "lost": lost, "samples_per_packet": spp}
        return audio, stats
    

