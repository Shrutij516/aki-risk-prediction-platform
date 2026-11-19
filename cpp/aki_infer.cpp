// aki_infer: C++ inference for an exported AKI risk model (AOTInductor package, LibTorch runtime).
//
// Usage: aki_infer <model.pt2> <inputs.txt> [batch_size=256]
//   inputs.txt : one patient per line, whitespace-separated int64 token ids, all lines the
//                same length (the model's maxlen, already encoded + padded by Python).
//   stdout     : one probability per line, sigmoid(logit), printed with 9 significant digits.
//   stderr     : timing summary.
//
// Scope: numeric inference only. Vocabulary lookup, truncation and padding are done in Python
// (aki.serve.export_cpp / aki.serve.registry); this program does not read clinical codes.
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::vector<int64_t>> read_matrix(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<std::vector<int64_t>> rows;
  std::string line;
  size_t width = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::vector<int64_t> row;
    int64_t v;
    while (ss >> v) row.push_back(v);
    if (width == 0) width = row.size();
    if (row.size() != width || width == 0)
      throw std::runtime_error("line " + std::to_string(rows.size() + 1) + " has " + std::to_string(row.size()) +
                               " ids, expected " + std::to_string(width));
    rows.push_back(std::move(row));
  }
  if (rows.empty()) throw std::runtime_error("no input rows in " + path);
  return rows;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: aki_infer <model.pt2> <inputs.txt> [batch_size=256]\n";
    return 2;
  }
  try {
    const std::string model_path = argv[1];
    const int64_t batch = argc > 3 ? std::stoll(argv[3]) : 256;
    if (batch < 1) throw std::runtime_error("batch_size must be >= 1");
    c10::InferenceMode guard;

    auto t0 = std::chrono::steady_clock::now();
    torch::inductor::AOTIModelPackageLoader loader(model_path);
    auto t1 = std::chrono::steady_clock::now();
    const auto meta = loader.get_metadata();
    const auto it = meta.find("aki_maxlen");
    if (it == meta.end()) throw std::runtime_error("package has no aki_maxlen metadata; re-export with aki.serve.export_cpp");
    const int64_t maxlen = std::stoll(it->second);
    const auto mb = meta.find("aki_max_batch");
    const int64_t max_batch = mb == meta.end() ? 1 : std::stoll(mb->second);
    if (batch > max_batch)
      throw std::runtime_error("batch_size " + std::to_string(batch) + " exceeds the exported limit " +
                               std::to_string(max_batch));
    const auto rows = read_matrix(argv[2]);
    const int64_t n = static_cast<int64_t>(rows.size()), width = static_cast<int64_t>(rows[0].size());
    if (width != maxlen)
      throw std::runtime_error("inputs have " + std::to_string(width) + " ids per row; model expects " +
                               std::to_string(maxlen));

    for (int64_t start = 0; start < n; start += batch) {
      const int64_t b = std::min(batch, n - start);
      auto ids = torch::empty({b, width}, torch::kLong);
      auto acc = ids.accessor<int64_t, 2>();
      for (int64_t i = 0; i < b; ++i)
        for (int64_t j = 0; j < width; ++j) acc[i][j] = rows[start + i][j];
      std::vector<at::Tensor> out = loader.run({ids});
      auto probs = torch::sigmoid(out.at(0)).contiguous();
      auto p = probs.accessor<float, 1>();
      for (int64_t i = 0; i < b; ++i) std::printf("%.9g\n", static_cast<double>(p[i]));
    }
    auto t2 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::fprintf(stderr, "aki_infer: %lld rows x %lld ids, load %.1f ms, read+infer %.1f ms (batch %lld)\n",
                 static_cast<long long>(n), static_cast<long long>(width), ms(t0, t1), ms(t1, t2),
                 static_cast<long long>(batch));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "aki_infer error: " << e.what() << "\n";
    return 1;
  }
}
