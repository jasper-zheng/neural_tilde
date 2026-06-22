#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>

namespace {
void put_u32(std::ofstream &f, uint32_t v) {
  char b[4] = {(char)(v & 0xff), (char)((v >> 8) & 0xff),
               (char)((v >> 16) & 0xff), (char)((v >> 24) & 0xff)};
  f.write(b, 4);
}
void put_u16(std::ofstream &f, uint16_t v) {
  char b[2] = {(char)(v & 0xff), (char)((v >> 8) & 0xff)};
  f.write(b, 2);
}
uint32_t get_u32(const unsigned char *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}
uint16_t get_u16(const unsigned char *p) {
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}
} // namespace

bool write_wav(const std::string &path, const std::vector<float> &audio,
               int channels, int length, int sample_rate) {
  if (channels <= 0 || length <= 0 ||
      (size_t)channels * (size_t)length > audio.size()) {
    std::cerr << "neural: write_wav bad dims (ch=" << channels
              << " len=" << length << " have=" << audio.size() << ")\n";
    return false;
  }
  for (size_t i = 0; i < (size_t)channels * length; i++)
    if (!std::isfinite(audio[i])) {
      std::cerr << "neural: refusing to write WAV — non-finite samples\n";
      return false;
    }

  // Interleave channel-major float -> int16 (L0,R0,L1,R1,...), clip to [-1,1].
  std::vector<int16_t> pcm((size_t)channels * length);
  for (int t = 0; t < length; t++)
    for (int c = 0; c < channels; c++) {
      float v = audio[(size_t)c * length + t];
      v = std::min(1.0f, std::max(-1.0f, v));
      pcm[(size_t)t * channels + c] = (int16_t)std::lround(v * 32767.0f);
    }

  const uint32_t data_bytes = (uint32_t)(pcm.size() * sizeof(int16_t));
  const uint16_t bits = 16;
  const uint16_t block_align = (uint16_t)(channels * bits / 8);
  const uint32_t byte_rate = (uint32_t)sample_rate * block_align;

  std::ofstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "neural: could not open " << path << " for writing\n";
    return false;
  }
  f.write("RIFF", 4);
  put_u32(f, 36 + data_bytes);
  f.write("WAVE", 4);
  f.write("fmt ", 4);
  put_u32(f, 16);                  // PCM fmt chunk size
  put_u16(f, 1);                   // PCM
  put_u16(f, (uint16_t)channels);
  put_u32(f, (uint32_t)sample_rate);
  put_u32(f, byte_rate);
  put_u16(f, block_align);
  put_u16(f, bits);
  f.write("data", 4);
  put_u32(f, data_bytes);
  f.write(reinterpret_cast<const char *>(pcm.data()), data_bytes);
  return (bool)f;
}

bool read_wav(const std::string &path, std::vector<float> &audio, int &channels,
              int &frames, int &sample_rate) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "neural: could not open " << path << " for reading\n";
    return false;
  }
  std::streamsize size = f.tellg();
  f.seekg(0);
  std::vector<unsigned char> buf((size_t)size);
  if (size < 44 || !f.read(reinterpret_cast<char *>(buf.data()), size)) {
    std::cerr << "neural: " << path << " is too small to be a WAV\n";
    return false;
  }
  if (std::memcmp(buf.data(), "RIFF", 4) != 0 ||
      std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
    std::cerr << "neural: " << path << " is not a RIFF/WAVE file\n";
    return false;
  }

  // Walk the chunk list to find "fmt " and "data" (chunks may be in any order
  // and there may be extra chunks, e.g. LIST/fact, between them).
  uint16_t fmt = 0, ch = 0, bits = 0;
  uint32_t sr = 0;
  const unsigned char *data = nullptr;
  size_t data_bytes = 0;
  size_t pos = 12;
  while (pos + 8 <= buf.size()) {
    const unsigned char *hdr = buf.data() + pos;
    uint32_t csize = get_u32(hdr + 4);
    const unsigned char *body = hdr + 8;
    if (pos + 8 + csize > buf.size())
      csize = (uint32_t)(buf.size() - pos - 8); // tolerate a truncated tail
    if (std::memcmp(hdr, "fmt ", 4) == 0 && csize >= 16) {
      fmt = get_u16(body);
      ch = get_u16(body + 2);
      sr = get_u32(body + 4);
      bits = get_u16(body + 14);
    } else if (std::memcmp(hdr, "data", 4) == 0) {
      data = body;
      data_bytes = csize;
    }
    pos += 8 + csize + (csize & 1); // chunks are word-aligned (pad byte)
  }

  if (!data || ch == 0 || sr == 0) {
    std::cerr << "neural: " << path << " missing fmt/data chunk\n";
    return false;
  }
  // fmt: 1 = PCM, 3 = IEEE float, 0xFFFE = extensible (treat per bit depth).
  const bool is_float = (fmt == 3) || (fmt == 0xFFFE && bits == 32);
  const bool is_pcm16 = (fmt == 1 || fmt == 0xFFFE) && bits == 16;
  if (!is_float && !is_pcm16) {
    std::cerr << "neural: " << path << " has unsupported format (fmt=" << fmt
              << ", bits=" << bits << "); only PCM16 and float32 are supported\n";
    return false;
  }

  const size_t bytes_per_sample = is_float ? 4 : 2;
  const size_t frame_bytes = bytes_per_sample * ch;
  const size_t n_frames = frame_bytes ? data_bytes / frame_bytes : 0;
  channels = (int)ch;
  frames = (int)n_frames;
  sample_rate = (int)sr;

  // De-interleave into channel-major float (all of ch0, then ch1, …).
  audio.assign((size_t)ch * n_frames, 0.0f);
  for (size_t t = 0; t < n_frames; t++) {
    for (uint16_t c = 0; c < ch; c++) {
      const unsigned char *s = data + (t * ch + c) * bytes_per_sample;
      float v;
      if (is_float) {
        uint32_t u = get_u32(s);
        std::memcpy(&v, &u, sizeof(float));
      } else {
        v = (int16_t)get_u16(s) / 32768.0f;
      }
      audio[(size_t)c * n_frames + t] = v;
    }
  }
  return true;
}
