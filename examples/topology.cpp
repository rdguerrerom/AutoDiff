#include "uff.hpp"

namespace UFF {

// Bond detection with correct bond order correction for fullerenes
std::vector<std::array<size_t, 2>> TopologyBuilder::detect_bonds(
    const std::vector<Atom>& atoms, 
    double tolerance  
) {
    std::vector<std::array<size_t, 2>> bonds;
    for(size_t i = 0; i < atoms.size(); ++i) {
        for(size_t j = i+1; j < atoms.size(); ++j) {
            const auto& a = atoms[i];
            const auto& b = atoms[j];

            // Get bond length parameters
            const double ri = Parameters::atom_parameters.at(a.type).r1;
            const double rj = Parameters::atom_parameters.at(b.type).r1;

            // Determine bond order for fullerene C60
            double bond_order = 1.0;
            if (a.type == "C_R" && b.type == "C_R") {
                bond_order = 1.5; // Intermediate bond order for resonance structures
            }

            // Pauling bond order correction
            constexpr double lambda = 0.1332;
            double rBO = -lambda * (ri + rj) * std::log(bond_order);

            // O'Keefe and Breese electronegativity correction
            const double Xi = Parameters::atom_parameters.at(a.type).GMP_Xi;
            const double Xj = Parameters::atom_parameters.at(b.type).GMP_Xi;
            const double rEN = ri * rj * (std::sqrt(Xi) - std::sqrt(Xj)) * (std::sqrt(Xi) - std::sqrt(Xj)) /
                (Xi * ri + Xj * rj);

            // Calculate natural bond length with corrections
            const double r_cov = ri + rj + rBO - rEN;

            // Calculate actual distance
            const double r_act = (a.position - b.position).norm();

            // Create bond if within tolerance
            if(r_act <= tolerance * r_cov) {
                bonds.push_back({i, j});
            }
        }
    }
    return bonds;
}

// Angle detection method
std::vector<std::array<size_t, 3>> TopologyBuilder::detect_angles(
    const std::vector<std::array<size_t, 2>>& bonds
) {
    std::unordered_map<size_t, std::vector<size_t>> adj_list;
    for(const auto& bond : bonds) {
        adj_list[bond[0]].push_back(bond[1]);
        adj_list[bond[1]].push_back(bond[0]);
    }

    std::vector<std::array<size_t, 3>> angles;
    for(const auto& [center, neighbors] : adj_list) {
        for(size_t i = 0; i < neighbors.size(); ++i) {
            for(size_t j = i+1; j < neighbors.size(); ++j) {
                angles.push_back({neighbors[i], center, neighbors[j]});
            }
        }
    }
    return angles;
}

// Torsion detection with proper deduplication
std::vector<std::array<size_t, 4>> TopologyBuilder::detect_torsions(
    const std::vector<std::array<size_t, 2>>& bonds) {
    std::unordered_map<size_t, std::vector<size_t>> adj_list;
    for (const auto& bond : bonds) {
        adj_list[bond[0]].push_back(bond[1]);
        adj_list[bond[1]].push_back(bond[0]);
    }

    // Use a standard set approach to avoid hash function issues
    std::vector<std::array<size_t, 4>> torsions;

    // Track processed torsions using vectors for canonical representation
    std::set<std::vector<size_t>> processed;

    for (const auto& bond : bonds) {
        size_t a = bond[0];
        size_t b = bond[1];

        // Process each central bond once
        if (a > b) continue;

        for (auto c : adj_list[a]) {
            if (c == b) continue;
            for (auto d : adj_list[b]) {
                if (d == a || d == c) continue;

                // Create a canonical representation of the torsion for deduplication
                std::vector<size_t> canonical;
                if (c < d || (c == d && a < b)) {
                    canonical = {c, a, b, d};
                } else {
                    canonical = {d, b, a, c};
                }

                // Add only if we haven't seen this torsion before
                if (processed.insert(canonical).second) {
                    torsions.push_back({canonical[0], canonical[1], canonical[2], canonical[3]});
                }
            }
        }
    }

    return torsions;
}

// Inversion term detection (out-of-plane bending)
std::vector<std::array<size_t, 4>> TopologyBuilder::detect_inversions(
    const std::vector<Atom>& atoms,
    const std::vector<std::array<size_t, 2>>& bonds) {

    // Build adjacency list
    std::unordered_map<size_t, std::vector<size_t>> adj_list;
    for(const auto& bond : bonds) {
        adj_list[bond[0]].push_back(bond[1]);
        adj_list[bond[1]].push_back(bond[0]);
    }

    std::vector<std::array<size_t, 4>> inversions;

    // Iterate through all atoms
    for(size_t j = 0; j < atoms.size(); ++j) {
        const auto& central_atom = atoms[j];

        // Only consider sp2 carbons and elements that need improper terms
        if (central_atom.type != "C_R" && 
            central_atom.type != "N_R" && 
            central_atom.type != "O_R" && 
            central_atom.type != "S_R") {
            continue;
        }

        // Need exactly 3 neighbors for an improper term
        if (adj_list[j].size() != 3) {
            continue;
        }

        // Get neighbors
        const size_t i = adj_list[j][0];
        const size_t k = adj_list[j][1];
        const size_t l = adj_list[j][2];

        // Add improper term (atom i, j, k define a plane, atom l is out of plane)
        inversions.push_back({i, j, k, l});
    }

    return inversions;
}

} // namespace UFF
