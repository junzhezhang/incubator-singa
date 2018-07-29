/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#include "../include/singa/io/reader.h"
#include "../include/singa/io/writer.h"
#include "gtest/gtest.h"

const char* path_csv = "./textfile_test.csv";
using singa::io::TextFileReader;
using singa::io::TextFileWriter;
TEST(TextFileWriter, Create) {
  TextFileWriter writer;
  bool ret;
  ret = writer.Open(path_csv, singa::io::kCreate);
  EXPECT_EQ(true, ret);

  std::string key = "";
  std::string value = "This is a test for binfile io.";
  ret = writer.Write(key, value);
  EXPECT_EQ(true, ret);

  ret = writer.Write(key, value);
  EXPECT_EQ(true, ret);

  writer.Flush();
  writer.Close();
}

TEST(TextFileWriter, Append) {
  TextFileWriter writer;
  bool ret;
  ret = writer.Open(path_csv, singa::io::kAppend);
  EXPECT_EQ(true, ret);

  std::string key = "1";
  std::string value = "This is another test for binfile io.";
  ret = writer.Write(key, value);
  EXPECT_EQ(true, ret);

  key = "2";
  value = "This is another test for binfile io.";
  ret = writer.Write(key, value);
  EXPECT_EQ(true, ret);

  writer.Flush();
  writer.Close();
}

TEST(TextFileReader, Read) {
  TextFileReader reader;
  bool ret;
  ret = reader.Open(path_csv);
  EXPECT_EQ(true, ret);

  int cnt = reader.Count();
  EXPECT_EQ(4, cnt);

  std::string key, value;
  reader.Read(&key, &value);
  EXPECT_STREQ("0", key.c_str());
  EXPECT_STREQ("This is a test for binfile io.", value.c_str());

  reader.Read(&key, &value);
  EXPECT_STREQ("1", key.c_str());
  EXPECT_STREQ("This is a test for binfile io.", value.c_str());

  reader.Read(&key, &value);
  EXPECT_STREQ("2", key.c_str());
  EXPECT_STREQ("This is another test for binfile io.", value.c_str());

  reader.Read(&key, &value);
  EXPECT_STREQ("3", key.c_str());
  EXPECT_STREQ("This is another test for binfile io.", value.c_str());

  reader.Close();
}

TEST(TextFileReader, SeekToFirst) {
  TextFileReader reader;
  bool ret;
  ret = reader.Open(path_csv);
  EXPECT_EQ(true, ret);

  int cnt = reader.Count();
  EXPECT_EQ(4, cnt);

  std::string key, value;
  reader.Read(&key, &value);
  EXPECT_STREQ("0", key.c_str());
  EXPECT_STREQ("This is a test for binfile io.", value.c_str());

  reader.Read(&key, &value);
  EXPECT_STREQ("1", key.c_str());
  EXPECT_STREQ("This is a test for binfile io.", value.c_str());

  reader.SeekToFirst();
  reader.Read(&key, &value);
  EXPECT_STREQ("0", key.c_str());
  EXPECT_STREQ("This is a test for binfile io.", value.c_str());

  reader.Read(&key, &value);
  EXPECT_STREQ("1", key.c_str());
  EXPECT_STREQ("This is a test for binfile io.", value.c_str());

  reader.Read(&key, &value);
  EXPECT_STREQ("2", key.c_str());
  EXPECT_STREQ("This is another test for binfile io.", value.c_str());

  reader.Read(&key, &value);
  EXPECT_STREQ("3", key.c_str());
  EXPECT_STREQ("This is another test for binfile io.", value.c_str());

  reader.Close();
  remove(path_csv);
}
