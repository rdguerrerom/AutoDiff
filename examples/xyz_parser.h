#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include "uff_simulator.h"

class XYZParser {
public:
    static std::vector<Atom> parse(const std::string& filename) {
        std::ifstream file(filename);
        std::vector<Atom> atoms;
        std::string line;
        
        // Skip atom count and comment line
        std::getline(file, line);
        std::getline(file, line);
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string element;
            double x, y, z;
            ss >> element >> x >> y >> z;
            
            // Convert element to UFF atom type (simple mapping)
            std::string uff_type = element;
            if (element == "C") uff_type = "C_sp2";  // Assume sp² for fullerenes
            
            atoms.emplace_back(uff_type, Vec3(x, y, z));
        }
        return atoms;
    }

    // Generate bonds based on covalent radii
    static std::vector<UFFBond> generateBonds(const std::vector<Atom>& atoms, double tolerance = 1.2) {
        std::vector<UFFBond> bonds;
        for (size_t i = 0; i < atoms.size(); ++i) {
            for (size_t j = i + 1; j < atoms.size(); ++j) {
                double r_cov_i = uff_atom_params[atoms[i].type]["r_cov"];
                double r_cov_j = uff_atom_params[atoms[j].type]["r_cov"];
                double max_dist = tolerance * (r_cov_i + r_cov_j);
                
                double actual_dist = (atoms[i].position - atoms[j].position).norm();
                if (actual_dist <= max_dist) {
                    bonds.push_back({
                        static_cast<int>(i), static_cast<int>(j),
                        uff_atom_params[atoms[i].type]["r_cov"] + uff_atom_params[atoms[j].type]["r_cov"],
                        uff_atom_params[atoms[i].type]["D_e"],
                        uff_atom_params[atoms[i].type]["alpha"]
                    });
                }
            }
        }
        return bonds;
    }
};


