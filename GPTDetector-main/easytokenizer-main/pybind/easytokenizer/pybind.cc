/**
 * Copyright (c) 2022-present, Zejun Wang (wangzejunscut@126.com)
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <tokenizer.h>

namespace py = pybind11;

using Encoding  = std::map<std::string, std::vector<int>>;
using Encodings = std::map<std::string, std::vector<std::vector<int>>>;

PYBIND11_MAKE_OPAQUE(Encoding);
PYBIND11_MAKE_OPAQUE(Encodings);

PYBIND11_MODULE(easytokenizer, m) {
  m.doc() = "An efficient and easy-to-use tokenization toolkit.";
  
  py::bind_map<Encoding>(m, "Encoding");  
  py::bind_map<Encodings>(m, "Encodings"); 

  py::class_<tokenizer::BasicTokenizer> BasicTokenizerClass(m, "BasicTokenizer");
  BasicTokenizerClass.def(py::init<bool>(), py::arg("do_lower_case") = true);
  
  py::class_<tokenizer::Tokenizer>(m, "AutoTokenizer", BasicTokenizerClass)
    .def(py::init<const std::string&, bool, bool>(), "Init AutoTokenizer",
         py::arg("vocab_path"), py::arg("do_lower_case") = true,
         py::arg("codepoint_level") = true)
    
    .def("insert", (void (tokenizer::Tokenizer::*)(const std::string&))
        (&tokenizer::Tokenizer::insert), py::arg("token"))
    .def("insert", (void (tokenizer::Tokenizer::*)(const std::vector<std::string>&))
        (&tokenizer::Tokenizer::insert), py::arg("tokens"))
    
    .def("add_special_tokens", (void (tokenizer::Tokenizer::*)(const std::string&))
        (&tokenizer::Tokenizer::add_special_tokens), py::arg("token"))
    .def("add_special_tokens", (void (tokenizer::Tokenizer::*)(const std::vector<std::string>&))
        (&tokenizer::Tokenizer::add_special_tokens), py::arg("tokens"))
    
    .def("pad_token", &tokenizer::Tokenizer::pad_token)
    .def("cls_token", &tokenizer::Tokenizer::cls_token)
    .def("sep_token", &tokenizer::Tokenizer::sep_token)
    .def("unk_token", &tokenizer::Tokenizer::unk_token)
    .def("mask_token", &tokenizer::Tokenizer::mask_token)
    .def("get_token", &tokenizer::Tokenizer::get_token, py::arg("id"))
    
    .def("count", &tokenizer::Tokenizer::count, py::arg("token"))
    .def("size", &tokenizer::Tokenizer::size)
    .def("pad_id", &tokenizer::Tokenizer::pad_id)
    .def("cls_id", &tokenizer::Tokenizer::cls_id)
    .def("sep_id", &tokenizer::Tokenizer::sep_id)
    .def("unk_id", &tokenizer::Tokenizer::unk_id)
    .def("mask_id", &tokenizer::Tokenizer::mask_id)
    .def("get_id", &tokenizer::Tokenizer::get_id, py::arg("token"))
    
    .def("convert_ids_to_tokens", &tokenizer::Tokenizer::convert_ids_to_tokens, 
         py::arg("input_ids"))
    .def(
      "convert_tokens_to_ids",
      [](tokenizer::Tokenizer& m, const std::vector<std::string>& tokens, bool add_cls_sep = false) {
        return m.convert_tokens_to_ids(tokens, add_cls_sep);
      },
      py::arg("tokens"), py::arg("add_cls_sep") = false
    )
    
    .def(
      "tokenize",
      [](tokenizer::Tokenizer& m, const std::string& text) {
        return m.wordpiece_tokenize(text);       
      },
      py::arg("text")
    )
    
    .def(
      "encode",
      [](tokenizer::Tokenizer& m, const std::string& text, 
        bool add_cls_sep = true, bool truncation = true, int max_length = 512) {
        std::vector<int> input_ids;
        std::vector<int> attention_mask;
        std::vector<int> offsets;
        m.encode(text, input_ids, attention_mask, offsets,
          add_cls_sep, truncation, max_length);

        Encoding encoding;
        encoding["input_ids"] = std::move(input_ids);
        encoding["attention_mask"] = std::move(attention_mask);
        encoding["offsets"] = std::move(offsets);
        return encoding;       
      },
      py::arg("text"),
      py::arg("add_cls_sep") = true,
      py::arg("truncation") = true, 
      py::arg("max_length") = 512
    )
    
    .def(
      "encode",
      [](tokenizer::Tokenizer& m, const std::vector<std::string>& texts, int num_threads = 1, 
        bool add_cls_sep = true, bool padding = true, bool padding_to_max_length = false, 
        bool truncation = true, int max_length = 512) {
        std::vector<std::vector<int>> input_ids;
        std::vector<std::vector<int>> attention_mask;
        std::vector<std::vector<int>> offsets;
        m.encode(texts, input_ids, attention_mask, offsets, num_threads, add_cls_sep, padding, 
          padding_to_max_length, truncation, max_length);

        Encodings encodings;
        encodings["input_ids"] = std::move(input_ids);
        encodings["attention_mask"] = std::move(attention_mask);
        encodings["offsets"] = std::move(offsets);
        return encodings;       
      },
      py::arg("texts"),
      py::arg("num_threads") = 1,
      py::arg("add_cls_sep") = true, 
      py::arg("padding") = true,
      py::arg("padding_to_max_length") = false, 
      py::arg("truncation") = true,
      py::arg("max_length") = 512
    );
}
