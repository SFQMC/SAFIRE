#pragma once

#include <hdf5.h>
#include <nda/nda.hpp>
#include <nda/h5.hpp>
#include <utilities/check.hpp>
namespace sfqmc::afqmc {

namespace detail {

template<typename T>
auto collect_full_bins(std::vector<T> const& bins) {
  utils::check(bins.size() > 0, "collect_full_bins cannot be called without bins");
  long const nfull = long(bins.size()) - 1;
  
  if constexpr (nda::Array<T>) {
    auto shape = bins.front().shape();
    nda::array<std::remove_const_t<nda::get_value_t<T>>, nda::get_rank<T> + 1> result(nda::stdutil::front_append(shape, nfull));

    for(int i = 0; i < nfull; i++) {
      utils::check(bins[i].shape() == shape, "Accumulator contained incompatible shapes: {} != {}", bins[i].shape(), shape);
      result(i, nda::ellipsis{}) = bins[i]();
    }
    return result;
  } else {
    nda::vector<T> result(nfull);
    for(long i = 0; i < nfull; i++) {
      result[i] = bins[i];
    }
    return result;
  }
}

/// Create `name` as an empty chunked dataset whose first dimension is unlimited, so that later calls
/// can grow it and append to it. `sample` fixes the extents of the remaining dimensions. Complex
/// values get the trailing size-2 dimension and the "__complex__" marker that h5 uses itself.
template<nda::MemoryArray A>
void create_extendable_dataset(h5::group const& out, std::string const& name, A const& sample) {
  constexpr bool is_complex = nda::is_complex_v<nda::get_value_t<A>>;
  constexpr int rank        = A::rank + is_complex;

  std::array<hsize_t, rank> dims;
  for(int i = 0; i < A::rank; i++) {
    dims[i] = sample.shape()[i];
  }
  if constexpr(is_complex) {
    dims[A::rank] = 2;
  }

  // one chunk per flush, matching the granularity the appends will write at
  auto chunk = dims;
  for(auto& c : chunk) {
    c = std::max(c, hsize_t{1});
  }

  auto maxdims = dims;
  maxdims[0]   = H5S_UNLIMITED;
  dims[0]      = 0;

  h5::proplist cparms = H5Pcreate(H5P_DATASET_CREATE);
  utils::check(H5Pset_chunk(cparms, rank, chunk.data()) >= 0, "H5Pset_chunk failed for dataset '{}'", name);
  H5Pset_deflate(cparms, 1);

  h5::dataspace dspace = H5Screate_simple(rank, dims.data(), maxdims.data());
  h5::dataset dset =
      H5Dcreate2(out, name.c_str(), h5::hdf5_type<nda::get_value_t<A>>(), dspace, H5P_DEFAULT, cparms, H5P_DEFAULT);
  utils::check(dset.is_valid(), "could not create dataset '{}'", name);

  if constexpr(is_complex) {
    h5::h5_write_attribute(dset, "__complex__", "1");
  }
}

}

class AccumulatorBase {
public:
  virtual ~AccumulatorBase() = default;
  virtual void write(h5::group const& out) = 0;
};

/// Accumulator averages scalar or `nda::array` samples of arbitrary rank and shape into bins of size `binsize`.
/// The averaged bins can then be appended to an hdf5 dataset.
template<typename T>
class Accumulator : public AccumulatorBase {
public:
  Accumulator(long binsize) : binsize_{binsize} {}

  void add(T const& sample) {
    // the shape of a sample is only known here, so the first bin cannot be made in the ctor
    if(bins_.empty()) {
      bins_.emplace_back(0*sample);
    }
    bins_.back() += sample;
    current_bin_filling_++;

    if(current_bin_filling_ >= binsize_) {
      bins_.back() /= binsize_;
      
      bins_.emplace_back(0*sample);
      current_bin_filling_ = 0;
    }      
  }

  /// Write complete bins to out and discard them from memory. Keep only the unfinished bin.
  void write(h5::group const& out) override {
    if(bins_.empty()) {
      return;
    }

    auto full_bins = detail::collect_full_bins(bins_);
    if(full_bins.extent(0) == 0) {
      return;
    }    
    
    if(!out.has_key("bins")) {
      detail::create_extendable_dataset(out, "bins", full_bins);
    }

    h5::dataset dset = out.open_dataset("bins");

    h5::dataspace dspace = H5Dget_space(dset);
    int dset_rank = H5Sget_simple_extent_ndims(dspace);
    std::vector<hsize_t> dims(dset_rank), maxdims(dset_rank);
    H5Sget_simple_extent_dims(dspace, dims.data(), maxdims.data());

    long written_bins = static_cast<long>(dims[0]);
    long appended_bins = full_bins.extent(0);

    utils::check(maxdims[0] == H5S_UNLIMITED || maxdims[0] >= dims[0] + appended_bins, "dataset 'bins' cannot grow past its max extent");

    dims[0] += appended_bins;
    utils::check(H5Dset_extent(dset, dims.data()) >= 0, "H5Dset_extent failed");

    nda::h5_write(out, "bins", full_bins, std::make_tuple(nda::range(written_bins, written_bins + appended_bins), nda::ellipsis{}));

    std::swap(bins_.front(), bins_.back());
    bins_.resize(1);
  }
private:
  std::vector<T> bins_;
  long current_bin_filling_{};
  long binsize_{1};
};

}
