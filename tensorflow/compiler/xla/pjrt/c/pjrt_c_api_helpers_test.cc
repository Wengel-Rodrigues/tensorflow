/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_helpers.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status.h"

namespace pjrt {
namespace {

using ::testing::HasSubstr;

TEST(PjRtCApiHelperTest, ConvertValidPjRtValueType) {
  std::vector<int64_t> int64_list = {static_cast<int64_t>(1),
                                     static_cast<int64_t>(2)};
  absl::flat_hash_map<std::string, xla::PjRtValueType> original_cpp_map = {
      {"string", "v1"},
      {"int64", static_cast<int64_t>(1)},
      {"int64_list", int64_list},
      {"float", static_cast<float>(1.0)}};

  TF_ASSERT_OK_AND_ASSIGN(std::vector<PJRT_NamedValue> c_map,
                          ConvertToPjRtNamedValueList(original_cpp_map));
  auto converted_back_cpp_map =
      ConvertFromPjRtNamedValueList(c_map.data(), c_map.size());

  EXPECT_THAT(converted_back_cpp_map,
              testing::UnorderedElementsAreArray(original_cpp_map));
}

TEST(PjRtCApiHelperTest, ValidOptionNameAndPjRtValueTypeIndex) {
  const auto expected = absl::flat_hash_map<std::string, PJRT_NamedValue_Type>({
      {"string", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
      {"int64", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
  });
  absl::flat_hash_map<std::string, xla::PjRtValueType> valid_map = {
      {"string", "v1"}, {"int64", static_cast<int64_t>(1)}};

  TF_EXPECT_OK(ValidateCreateOptions(valid_map, expected));
}

TEST(PjRtCApiHelperTest, InvalidOptionName) {
  const auto expected = absl::flat_hash_map<std::string, PJRT_NamedValue_Type>({
      {"string", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
      {"int64", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
  });
  absl::flat_hash_map<std::string, xla::PjRtValueType> invalid_map = {
      {"invalid", "v1"}};

  auto status = ValidateCreateOptions(invalid_map, expected);

  EXPECT_NE(status, tsl::OkStatus());
  EXPECT_THAT(status.message(),
              HasSubstr("Unexpected option name passed to PJRT_Client_Create"));
}

TEST(PjRtCApiHelperTest, InvalidOptionTypeIndex) {
  const auto expected = absl::flat_hash_map<std::string, PJRT_NamedValue_Type>({
      {"string", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
      {"int64", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
  });
  absl::flat_hash_map<std::string, xla::PjRtValueType> invalid_map = {
      {"string", static_cast<int64_t>(1)}};

  auto status = ValidateCreateOptions(invalid_map, expected);

  EXPECT_NE(status, tsl::OkStatus());
  EXPECT_THAT(status.message(),
              HasSubstr("Option passed to PJRT_Client_Create with name string "
                        "has type index 1 but expected type index is 0"));
}

TEST(PjRtCApiHelperTest, Callback) {
  absl::flat_hash_map<std::string, std::string> kv_store;
  absl::Mutex mu;
  xla::PjRtClient::KeyValueGetCallback kv_get =
      [&kv_store, &mu](const std::string& k,
                       absl::Duration timeout) -> xla::StatusOr<std::string> {
    absl::Duration wait_interval = absl::Milliseconds(10);
    int num_retry = timeout / wait_interval;
    for (int i = 0; i < num_retry; i++) {
      {
        absl::MutexLock lock(&mu);
        auto iter = kv_store.find(k);
        if (iter != kv_store.end()) {
          return iter->second;
        }
      }
      absl::SleepFor(wait_interval);
    }
    return absl::NotFoundError(
        absl::StrCat(k, " is not found in the kv store."));
  };
  xla::PjRtClient::KeyValuePutCallback kv_put =
      [&kv_store, &mu](const std::string& k,
                       const std::string& v) -> xla::Status {
    {
      absl::MutexLock lock(&mu);
      kv_store[k] = v;
    }
    return tsl::OkStatus();
  };
  auto kv_callback_data = ConvertToCKeyValueCallbacks(kv_get, kv_put);
  auto converted_back_kv_get = ToCppKeyValueGetCallback(
      kv_callback_data->c_kv_get, &kv_callback_data->kv_get_c_func);
  auto converted_back_kv_put = ToCppKeyValuePutCallback(
      kv_callback_data->c_kv_put, &kv_callback_data->kv_put_c_func);

  auto s = converted_back_kv_put("key", "value");
  TF_EXPECT_OK(s);

  auto v = converted_back_kv_get("key", absl::Seconds(1));
  TF_EXPECT_OK(v.status());
  EXPECT_EQ(*v, "value");
}

TEST(PjRtCApiHelperTest, ConvertToCLayoutFromStrides) {
  std::vector<int64_t> strides = {4, 8};
  xla::StatusOr<BufferMemoryLayoutData> layout_data =
      ConvertToBufferMemoryLayoutData(strides);

  EXPECT_TRUE(layout_data.ok());
  EXPECT_EQ(
      layout_data->c_layout.type,
      PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Strides);
  EXPECT_EQ(layout_data->c_layout.strides.num_byte_strides, 2);
  EXPECT_EQ(layout_data->c_layout.strides.byte_strides[0], strides[0]);
  EXPECT_EQ(layout_data->c_layout.strides.byte_strides[1], strides[1]);
}

TEST(PjRtCApiHelperTest, ConvertToCLayoutFromLayoutNoTiles) {
  std::vector<int64_t> minor_to_major = {1, 0};
  xla::Layout layout(minor_to_major);

  TF_ASSERT_OK_AND_ASSIGN(BufferMemoryLayoutData layout_data,
                          ConvertToBufferMemoryLayoutData(&layout));

  EXPECT_EQ(layout_data.c_layout.type,
            PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled);
  EXPECT_EQ(layout_data.c_layout.tiled.num_tiles, 0);
  PJRT_Buffer_MemoryLayout_Tiled tiled = layout_data.c_layout.tiled;
  EXPECT_EQ(tiled.minor_to_major_size, 2);
  EXPECT_EQ(tiled.minor_to_major[0], minor_to_major[0]);
  EXPECT_EQ(tiled.minor_to_major[1], minor_to_major[1]);
}

TEST(PjRtCApiHelperTest, ConvertToCLayoutFromLayoutWithTiles) {
  std::vector<int64_t> minor_to_major = {1, 0};
  xla::Layout layout(minor_to_major);
  std::vector<int64_t> tile_dims_1 = {2, 4};
  std::vector<int64_t> tile_dims_2 = {1};
  layout.mutable_tiles()->push_back(xla::Tile(tile_dims_1));
  layout.mutable_tiles()->push_back(xla::Tile(tile_dims_2));

  TF_ASSERT_OK_AND_ASSIGN(BufferMemoryLayoutData layout_data,
                          ConvertToBufferMemoryLayoutData(&layout));

  EXPECT_EQ(layout_data.c_layout.type,
            PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled);
  PJRT_Buffer_MemoryLayout_Tiled tiled = layout_data.c_layout.tiled;
  EXPECT_EQ(tiled.minor_to_major_size, 2);
  EXPECT_EQ(tiled.minor_to_major[0], minor_to_major[0]);
  EXPECT_EQ(tiled.minor_to_major[1], minor_to_major[1]);
  EXPECT_EQ(tiled.num_tiles, 2);
  EXPECT_EQ(tiled.tile_dim_sizes[0], tile_dims_1.size());
  EXPECT_EQ(tiled.tile_dim_sizes[1], tile_dims_2.size());
  EXPECT_EQ(tiled.tile_dims[0], tile_dims_1[0]);
  EXPECT_EQ(tiled.tile_dims[1], tile_dims_1[1]);
  EXPECT_EQ(tiled.tile_dims[2], tile_dims_2[0]);
}

TEST(PjRtCApiHelperTest, ConvertFromCLayoutToLayout) {
  PJRT_Buffer_MemoryLayout c_layout;
  c_layout.type =
      PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled;
  std::vector<int64_t> minor_to_major = {1, 0};
  c_layout.tiled.minor_to_major_size = 2;
  c_layout.tiled.minor_to_major = minor_to_major.data();
  c_layout.tiled.num_tiles = 2;
  std::vector<size_t> tile_dim_sizes = {2, 1};
  c_layout.tiled.tile_dim_sizes = tile_dim_sizes.data();
  std::vector<int64_t> tile_dims = {2, 4, 1};
  c_layout.tiled.tile_dims = tile_dims.data();

  TF_ASSERT_OK_AND_ASSIGN(xla::Layout layout, ConvertToLayout(c_layout.tiled));

  EXPECT_EQ(layout.ToString(), "{1,0:T(2,4)(1)}");
}

TEST(PjRtCApiHelperTest, ConvertFromCLayoutToLayoutNoTile) {
  PJRT_Buffer_MemoryLayout c_layout;
  c_layout.type =
      PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled;
  c_layout.tiled.num_tiles = 0;
  std::vector<int64_t> minor_to_major = {1, 0};
  c_layout.tiled.minor_to_major_size = 2;
  c_layout.tiled.minor_to_major = minor_to_major.data();

  TF_ASSERT_OK_AND_ASSIGN(xla::Layout layout, ConvertToLayout(c_layout.tiled));

  EXPECT_EQ(layout.ToString(), "{1,0}");
}

}  // namespace
}  // namespace pjrt
