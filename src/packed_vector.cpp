/*
 * Copyright (c) 2012 Alexander Sviridenko
 */
#include <cstdlib>
#include <assert.h>
#include <string>
#include <iostream>
#include <vector>

#include <les/packed_vector.hpp>

#include <coin/CoinHelperFunctions.hpp>

#define max(a, b) ((a) > (b)) ? (a) : (b)

using namespace std;

PackedVector::PackedVector()
{
  indices_ = NULL;
  elements_ = NULL;
  nr_elements_ = 0;
  orig_indices_ = NULL;
  capacity_ = 0;
}

void
PackedVector::clear()
{
  nr_elements_ = 0;
}

void
PackedVector::zero()
{
  for (int i = 0; i < get_num_elements(); i++)
    elements_[i] = 0.0;
  //CoinIotaN(elements_, nr_elements_, 0.0);
}

void
PackedVector::init(size_t size, const int* indices, double* elements)
{
  clear();
  if (size > 0)
    {
      reserve(size);
      nr_elements_ = size;
      CoinDisjointCopyN(indices, size, indices_);
      if (elements != NULL)
        CoinDisjointCopyN(elements, size, elements_);
      else
        zero();
      CoinIotaN(orig_indices_, size, 0);
    }
}

void
PackedVector::init(vector<int>& indices)
{
  init(indices.size(), &indices[0], NULL);
}

void
PackedVector::init(vector<int>& indices, vector<double>& elements)
{
  init(indices.size(), &indices[0], &elements[0]);
}

void
PackedVector::insert(int index, double element)
{
  if(capacity_ <= get_num_elements())
    {
      reserve(max(5, 2 * capacity_));
      assert(capacity_ > get_num_elements());
    }
  indices_[get_num_elements()] = index;
  index_to_pos_mapping_[index] = get_num_elements();
  elements_[get_num_elements()] = element;
  orig_indices_[get_num_elements()] = get_num_elements();
  nr_elements_++;
}

void
PackedVector::reserve(int n)
{
  /* don't make allocated space smaller */
  if (n <= capacity_)
    return;

  capacity_ = n;

  /* save pointers to existing data */
  int* temp_indices = indices_;
  int* temp_orig_indices = orig_indices_;
  double* temp_elements = elements_;

  /* allocate new space */
  indices_ = new int [capacity_];
  orig_indices_ = new int [capacity_];
  elements_ = new double [capacity_];

  /* copy data to new space */
  if (get_num_elements() > 0)
    {
      memcpy(indices_, temp_indices, get_num_elements());
      memcpy(orig_indices_, temp_orig_indices, get_num_elements());
      memcpy(elements_, temp_elements, get_num_elements());
    }

  /* free old data */
  delete [] temp_elements;
  delete [] temp_orig_indices;
  delete [] temp_indices;
}
