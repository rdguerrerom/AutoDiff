#pragma once

#include <unordered_map>
#include <string>

// LJ parameters for UFF atom types (σ in Å, ε in kcal/mol)
std::unordered_map<std::string, std::unordered_map<std::string, double>> uff_atom_params = {
    // sp² carbon (C₆₀)
    {"C_sp2", {
        {"sigma", 3.851}, {"epsilon", 0.105}, {"charge", 0.0},
        {"r_cov", 0.732}, {"alpha", 2.0}, {"k_theta", 63.134}
    }},
    // Noble gases
    {"He", {{"sigma", 2.600}, {"epsilon", 0.022}, {"charge", 0.0}}},
    {"Ne", {{"sigma", 3.190}, {"epsilon", 0.049}, {"charge", 0.0}}},
    {"Ar", {{"sigma", 3.780}, {"epsilon", 0.238}, {"charge", 0.0}}},
    {"Kr", {{"sigma", 4.050}, {"epsilon", 0.361}, {"charge", 0.0}}},
    {"Xe", {{"sigma", 4.380}, {"epsilon", 0.588}, {"charge", 0.0}}}
};

// Custom C-X LJ parameters (overrides mixing rules)
std::unordered_map<std::string, std::unordered_map<std::string, std::pair<double, double>>> custom_cx_pairs = {
    {"C_sp2", {
        {"He", {3.20, 0.030}},  // σ, ε
        {"Ar", {3.60, 0.150}},
        {"Xe", {3.95, 0.300}}
    }}
};

