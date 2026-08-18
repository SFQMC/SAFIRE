#include "parameters.hpp"

#include "IO/banner.hpp"

namespace sfqmc::afqmc {

namespace {

/// The values start in a column that fits the longest key of the input schema, so that the layout of
/// a block does not depend on which parameters it happens to contain. A key that does not fit is
/// followed by a single space instead.
constexpr int key_width = static_cast<int>(std::string_view{"measure_interval_multiplier     "}.size());

/// A value that gets lines of its own: an object, or an array that holds objects or arrays. Anything
/// else, including an empty object and an array of plain values, is printed after its key.
bool is_block(const nlohmann::ordered_json& json) {
  if(json.is_object()) {
    return !json.empty();
  }
  if(json.is_array()) {
    return std::ranges::any_of(json, [](const auto& value) { return value.is_object() || value.is_array(); });
  }
  return false;
}

std::string inline_value(const nlohmann::ordered_json& json) {
  if(!json.is_array()) {
    return json.dump();
  }
  std::string list;
  for(const auto& value : json) {
    list += std::format("{}{}", list.empty() ? "" : ", ", value.dump());
  }
  return std::format("[{}]", list);
}

void new_line(std::string& buf, int column) {
  buf += '\n';
  buf.append(column, ' ');
}

/// Appends the members of the block json, one per line, indented to column, in the yaml style: a
/// member that is a block itself starts on the line below its key, indented by one more level, and
/// the elements of an array are introduced by a dash. The marker goes in front of the first line, in
/// the columns to the left of column, which is how an element continues on the line of its dash.
void append_block(std::string& buf, const nlohmann::ordered_json& json, int column, int value_column,
                  std::string_view marker = {}) {
  bool first{true};
  const auto begin_line = [&] {
    new_line(buf, column - (first ? static_cast<int>(marker.size()) : 0));
    if(first) {
      buf += marker;
      first = false;
    }
  };

  if(json.is_object()) {
    for(const auto& [key, value] : json.items()) {
      begin_line();
      if(is_block(value)) {
        buf += std::format("{}:", key);
        append_block(buf, value, column + 2, value_column);
      } else {
        const int width = std::max(value_column - column - 1, static_cast<int>(key.size()) + 1);
        buf += std::format("{:<{}} {}", std::format("{}:", key), width, inline_value(value));
      }
    }
  } else {
    for(const auto& value : json) {
      if(!is_block(value)) {
        begin_line();
        buf += std::format("- {}", inline_value(value));
        continue;
      }
      if(!first) {
        new_line(buf, 0); // an empty line between the elements
      }
      const std::string element_marker = std::format("{}- ", first ? marker : std::string_view{});
      first = false;
      append_block(buf, value, column + 2, value_column, element_marker);
    }
  }
}

/// Formats json as a block of "key: value" lines, all of them indented to column.
std::string parameter_string(const nlohmann::ordered_json& json, int column) {
  std::string buf;
  append_block(buf, json, column, key_width + 4);
  if(!buf.empty()) {
    buf.erase(0, 1); // the newline in front of the first line
  }
  return buf;
}

} // namespace

void print_parameters(const AFQMCParameters& params) {
  app_log(2, banner("Input parameters"));

  app_log(2, section("Project"));
  app_log(2, parameter_string(nlohmann::ordered_json(params.project), 2));

  int n{};
  for(const ExecuteParameters& exec : params.execute) {
    app_log(2, section(std::format("Execution Stage {}", n)));
    app_log(2, parameter_string(nlohmann::ordered_json(exec), 2));
    n++;
  }

  app_log(2, section("Hamiltonians"));
  app_log(2, parameter_string(nlohmann::ordered_json(params.hamiltonian), 0));
  
  app_log(2, section("Wavefunctions"));
  app_log(2, parameter_string(nlohmann::ordered_json(params.wavefunction), 0));
  
  app_log(2, section("Walker Sets"));
  app_log(2, parameter_string(nlohmann::ordered_json(params.walker_set), 0));
  
  app_log(2, section("Propagators"));
  app_log(2, parameter_string(nlohmann::ordered_json(params.propagator), 0));
}


}
