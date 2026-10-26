#pragma once

#include <string>
#include <vector>

#include "common/runner.h"

// Append one row to results.csv (writing the header first if the file is new or empty).
// Creates parent directories. Columns follow docs/04-data-schema/DATA_SCHEMA.md.
void append_result(const std::string& path, const Row& row);

// Append every repetition of a run to reps.csv (same directory, same rules).
void append_reps(const std::string& path, const std::string& run_id, const std::vector<Timing>& reps);

// "2026-10-26T10:15:00Z" for the current time, and a unique id for a run.
std::string iso_timestamp_now();
std::string make_run_id();
