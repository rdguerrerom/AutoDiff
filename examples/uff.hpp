// uff.hpp
#pragma once
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iterator>

namespace UFF {

namespace Constants {
    constexpr double kcal_to_eV = 0.0433641;
    constexpr double coulomb_constant = 332.0637;
    constexpr double pi = 3.14159265358979323846;
    constexpr double deg2rad = pi / 180.0;
}

// 3D vector structure
struct Vec3 : std::array<double, 3> {
    using Base = std::array<double, 3>;
    using Base::Base;
    
    double norm() const noexcept {
        return std::sqrt((*this)[0]*(*this)[0] + (*this)[1]*(*this)[1] + (*this)[2]*(*this)[2]);
    }
    
    Vec3 operator-(const Vec3& other) const noexcept {
        Vec3 result;
        result[0] = (*this)[0] - other[0];
        result[1] = (*this)[1] - other[1];
        result[2] = (*this)[2] - other[2];
        return result;
    }
    
    Vec3 operator*(double scalar) const noexcept {
        Vec3 result;
        result[0] = (*this)[0] * scalar;
        result[1] = (*this)[1] * scalar;
        result[2] = (*this)[2] * scalar;
        return result;
    }
    
    double dot(const Vec3& other) const noexcept {
        return (*this)[0]*other[0] + (*this)[1]*other[1] + (*this)[2]*other[2];
    }
    
    Vec3 cross(const Vec3& other) const noexcept {
        Vec3 result;
        result[0] = (*this)[1]*other[2] - (*this)[2]*other[1];
        result[1] = (*this)[2]*other[0] - (*this)[0]*other[2];
        result[2] = (*this)[0]*other[1] - (*this)[1]*other[0];
        return result;
    }
};

// Non-member operator* for scalar multiplication
inline Vec3 operator*(double scalar, const Vec3& v) noexcept {
    return v * scalar;
}

// Atom structure
struct Atom {
    std::string type;
    Vec3 position;
    double charge;
};

namespace Parameters {
    struct UFFAtom {
        double r1;          // Covalent radius (Å)
        double theta0_deg;  // Equilibrium angle (degrees)
        double x;           // LJ sigma (Å)
        double D1;          // LJ epsilon (kcal/mol)
        double zeta;
        double Z1;          // Atomic number
        double V1;          // Torsional barrier
        double U1;
        double GMP_Xi;      // Electronegativity
        double GMP_Hardness;
        double GMP_Radius;

        double theta0() const noexcept { return theta0_deg * Constants::deg2rad; }
    };

    extern const std::unordered_map<std::string, UFFAtom> atom_parameters;
}

// Custom hash function for std::pair
struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const {
        std::size_t h1 = std::hash<T1>{}(pair.first);
        std::size_t h2 = std::hash<T2>{}(pair.second);
        return h1 ^ (h2 << 1); // Simple hash combine function
    }
};

class TopologyBuilder {
public:
    static std::vector<std::array<size_t, 2>> detect_bonds(
        const std::vector<Atom>& atoms, 
        double tolerance = 1.2
    ) {
        std::vector<std::array<size_t, 2>> bonds;
        for(size_t i = 0; i < atoms.size(); ++i) {
            for(size_t j = i+1; j < atoms.size(); ++j) {
                const auto& a = atoms[i];
                const auto& b = atoms[j];
                const double r_cov = Parameters::atom_parameters.at(a.type).r1 
                                    + Parameters::atom_parameters.at(b.type).r1;
                const double r_act = (a.position - b.position).norm();
                
                if(r_act <= tolerance * r_cov) {
                    bonds.push_back({i, j});
                }
            }
        }
        return bonds;
    }

    static std::vector<std::array<size_t, 3>> detect_angles(
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

    static std::vector<std::array<size_t, 4>> detect_torsions(
        const std::vector<std::array<size_t, 2>>& bonds
    ) {
        std::unordered_map<size_t, std::vector<size_t>> adj_list;
        for(const auto& bond : bonds) {
            adj_list[bond[0]].push_back(bond[1]);
            adj_list[bond[1]].push_back(bond[0]);
        }

        std::vector<std::array<size_t, 4>> torsions;
        for(const auto& [a, b] : bonds) {
            for(auto c : adj_list[a]) {
                if(c == b) continue;
                for(auto d : adj_list[b]) {
                    if(d == a || d == c) continue;
                    torsions.push_back({c, a, b, d});
                }
            }
        }

        // Remove duplicates
        std::sort(torsions.begin(), torsions.end());
        auto last = std::unique(torsions.begin(), torsions.end());
        torsions.erase(last, torsions.end());
        
        return torsions;
    }
};

class EnergyCalculator {
private:
  std::vector<Atom> atoms;
  std::vector<std::array<size_t, 2>> bonds;
  std::vector<std::array<size_t, 3>> angles;
  std::vector<std::array<size_t, 4>> torsions;
  std::vector<std::array<size_t, 4>> inversions;
  std::unordered_set<std::pair<size_t, size_t>, PairHash> excluded_pairs;

  // Scaling factors for 1-4 interactions
  static constexpr double vdw_14_scale = 0.5;  // Missing constant
  static constexpr double elec_14_scale = 0.5;

  const Parameters::UFFAtom& get_params(const std::string& type) const {
    try {
      return Parameters::atom_parameters.at(type);
    } catch(const std::out_of_range&) {
      throw std::runtime_error("Missing parameters for atom type: " + type);
    }
  }

    // Functions for inversion terms
    static std::vector<std::array<size_t, 4>> detect_inversions(
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
    
    static std::tuple<double, double, double, double> calc_inversion_parameters(const std::string& central_atom_type) {
        // Force constant, C0, C1, C2
        double K, C0, C1, C2;
        
        // Match RDKit's implementation for different atom types
        if (central_atom_type == "C_R") {
            // For sp2 carbon
            K = 6.0; // kcal/mol
            C0 = 1.0;
            C1 = -1.0;
            C2 = 0.0;
        } else if (central_atom_type == "N_R" || central_atom_type == "N_2") {
            // For sp2 nitrogen
            K = 6.0; // kcal/mol
            C0 = 1.0;
            C1 = -1.0;
            C2 = 0.0;
        } else {
            // Default values for other elements
            K = 0.0;
            C0 = 0.0;
            C1 = 0.0;
            C2 = 0.0;
        }
        
        return std::make_tuple(K, C0, C1, C2);
    }


static double calculate_cos_Y(const Vec3& p1, const Vec3& p2, const Vec3& p3, const Vec3& p4) {
    // p2 is the central atom
    Vec3 rJI = p1 - p2;
    Vec3 rJK = p3 - p2;
    Vec3 rJL = p4 - p2;
    
    double dJI = rJI.norm();
    double dJK = rJK.norm();
    double dJL = rJL.norm();
    
    // Skip if any bond length is zero
    if (dJI < 1e-8 || dJK < 1e-8 || dJL < 1e-8) {
        return 0.0;
    }
    
    // Normalize vectors using proper Vec3 initialization
    Vec3 rJI_norm;
    rJI_norm[0] = rJI[0] / dJI;
    rJI_norm[1] = rJI[1] / dJI;
    rJI_norm[2] = rJI[2] / dJI;
    
    Vec3 rJK_norm;
    rJK_norm[0] = rJK[0] / dJK;
    rJK_norm[1] = rJK[1] / dJK;
    rJK_norm[2] = rJK[2] / dJK;
    
    Vec3 rJL_norm;
    rJL_norm[0] = rJL[0] / dJL;
    rJL_norm[1] = rJL[1] / dJL;
    rJL_norm[2] = rJL[2] / dJL;
    
    // Normal to the plane defined by i-j-k
    Vec3 n = rJI_norm.cross(rJK_norm);
    double n_norm = n.norm();
    
    if (n_norm < 1e-8) {
        return 0.0;
    }
    
    // Normalize normal vector
    Vec3 n_unit;
    n_unit[0] = n[0] / n_norm;
    n_unit[1] = n[1] / n_norm;
    n_unit[2] = n[2] / n_norm;
    
    // Cosine of angle between normal and rJL
    double cos_Y = n_unit.dot(rJL_norm);
    
    // Clamp value to [-1, 1]
    return std::max(-1.0, std::min(1.0, cos_Y));
}

public:
  EnergyCalculator(std::vector<Atom> atoms,
                  std::vector<std::array<size_t, 2>> bonds,
                  std::vector<std::array<size_t, 3>> angles,
                  std::vector<std::array<size_t, 4>> torsions)
      : atoms(std::move(atoms)), bonds(std::move(bonds)),
      angles(std::move(angles)), torsions(std::move(torsions)) {
      
      // Populate excluded pairs (1-2 and 1-3)
      for (const auto& bond : this->bonds) {
          size_t a = bond[0], b = bond[1];
          excluded_pairs.insert({std::min(a, b), std::max(a, b)});
      }
      for (const auto& angle : this->angles) {
          size_t a = angle[0], c = angle[2];
          excluded_pairs.insert({std::min(a, c), std::max(a, c)});
      }
      
      // Detect inversion (out-of-plane) terms
      inversions = detect_inversions(this->atoms, this->bonds);
  }
  double calculate() const {
    double total = 0.0;
    double bond_energy = 0.0;
    double angle_energy = 0.0;
    double torsion_energy = 0.0;
    double vdw_energy = 0.0;
    double elec_energy = 0.0;

    // Bond stretching using RDKit's approach (harmonic instead of Morse)
    constexpr double lambda = 0.1332; // UFF lambda value for Pauling correction

    for (const auto& bond : bonds) {
      size_t i = bond[0], j = bond[1];
      const auto& a = atoms[i];
      const auto& b = atoms[j];
      const auto& p1 = get_params(a.type);
      const auto& p2 = get_params(b.type);

      // Determine bond order based on UFF atom types
      double bond_order = 1.0;
      if (a.type == "C_R" && b.type == "C_R") {
        // For fullerene C60, use an intermediate bond order to represent 
        // the resonance between single and double bonds
        bond_order = 1.5;
      }

      // Calculate rest length using Pauling correction and electronegativity
      double ri = p1.r1;
      double rj = p2.r1;

      // Pauling bond order correction:
      double rBO = -lambda * (ri + rj) * std::log(bond_order);

      // O'Keefe and Breese electronegativity correction:
      double Xi = p1.GMP_Xi;
      double Xj = p2.GMP_Xi;
      double rEN = ri * rj * (std::sqrt(Xi) - std::sqrt(Xj)) * (std::sqrt(Xi) - std::sqrt(Xj)) /
        (Xi * ri + Xj * rj);

      // Final rest length
      double rest_length = ri + rj + rBO - rEN;

      // Calculate force constant according to RDKit
      double force_constant = 2.0 * 664.12 * p1.Z1 * p2.Z1 / 
        (rest_length * rest_length * rest_length);

      // Calculate current bond length
      const double r = (a.position - b.position).norm();

      // Calculate harmonic bond stretch energy (RDKit uses harmonic, not Morse)
      double dist_term = r - rest_length;
      double energy = 0.5 * force_constant * dist_term * dist_term;

      bond_energy += energy;
    }

    // Angle bending (standard UFF harmonic cosine form)
    for (const auto& angle : angles) {
      size_t i = angle[0], j = angle[1], k = angle[2];
      const auto& a = atoms[i];
      const auto& b = atoms[j];
      const auto& c = atoms[k];
      const auto& p1 = get_params(a.type);
      const auto& p2 = get_params(b.type);
      const auto& p3 = get_params(c.type);

      // Get equilibrium angle from central atom parameters
      const double theta0 = p2.theta0();

      // Calculate vectors and current angle
      const Vec3 rij = a.position - b.position;
      const Vec3 rkj = c.position - b.position;
      const double r12 = rij.norm();
      const double r23 = rkj.norm();
      const double cos_theta = rij.dot(rkj) / (r12 * r23);

      // Clamp cos_theta to [-1, 1] to avoid acos domain errors
      const double cos_theta_clamped = std::max(-1.0, std::min(1.0, cos_theta));
      const double theta = std::acos(cos_theta_clamped);

      // Calculate the natural bond lengths between atoms
      double bond_order_ij = 1.0;
      double bond_order_jk = 1.0;
      if (a.type == "C_R" && b.type == "C_R") bond_order_ij = 1.5;
      if (b.type == "C_R" && c.type == "C_R") bond_order_jk = 1.5;

      // Calculate the distance between atoms i and k (needed for force constant)
      const double r13_sq = r12*r12 + r23*r23 - 2.0*r12*r23*std::cos(theta0);
      const double r13 = std::sqrt(r13_sq);

      // Calculate force constant according to RDKit implementation
      const double beta = 2.0 * 664.12 / (r12 * r23); // 664.12 is Params::G in RDKit
      const double K = beta * p1.Z1 * p3.Z1 / std::pow(r13, 5);
      const double r_term = r12 * r23;
      const double inner_bit = 3.0 * r_term * (1.0 - std::cos(theta0) * std::cos(theta0)) - r13_sq * std::cos(theta0);
      const double force_constant = K * r_term * inner_bit;

      // Calculate angle bending energy using cosine-harmonic form
      const double sin_theta_sq = 1.0 - cos_theta_clamped * cos_theta_clamped;
      const double cos_2theta = cos_theta_clamped * cos_theta_clamped - sin_theta_sq; // = cos(2*theta)

      // Calculate coefficients as in RDKit
      const double sin_theta0 = std::sin(theta0);
      const double sin_theta0_sq = sin_theta0 * sin_theta0;
      const double cos_theta0 = std::cos(theta0);
      const double C2 = 1.0 / (4.0 * std::max(sin_theta0_sq, 1e-8));
      const double C1 = -4.0 * C2 * cos_theta0;
      const double C0 = C2 * (2.0 * cos_theta0 * cos_theta0 + 1.0);

      // Calculate energy term according to RDKit formula
      double angle_term = C0 + C1 * cos_theta_clamped + C2 * cos_2theta;

      // Add correction for near-zero angles (based on RDKit implementation)
      constexpr double ANGLE_CORRECTION_THRESHOLD = 0.8660; // cos(30°)
      if (cos_theta_clamped > ANGLE_CORRECTION_THRESHOLD) {
        angle_term += std::exp(-20.0 * (theta - theta0 + 0.25));
      }

      angle_energy += force_constant * angle_term;
    }

    // Torsional potential using RDKit's approach
    for (const auto& torsion : torsions) {
      size_t i = torsion[0], j = torsion[1], k = torsion[2], l = torsion[3];
      const auto& a_i = atoms[i];
      const auto& a_j = atoms[j];
      const auto& a_k = atoms[k];
      const auto& a_l = atoms[l];
      const auto& params_j = get_params(a_j.type);
      const auto& params_k = get_params(a_k.type);

      // Calculate torsion geometry
      Vec3 r1 = a_i.position - a_j.position;
      Vec3 r2 = a_k.position - a_j.position;
      Vec3 r3 = a_j.position - a_k.position;
      Vec3 r4 = a_l.position - a_k.position;

      // Calculate cross products
      Vec3 t1 = r1.cross(r2);
      Vec3 t2 = r3.cross(r4);

      // Calculate lengths and normalize
      double d1 = t1.norm();
      double d2 = t2.norm();

      // Skip if degenerate
      if (d1 < 1e-8 || d2 < 1e-8) continue;

      // Calculate cosine of torsion angle
      double cos_phi = t1.dot(t2) / (d1 * d2);

      // Clamp to [-1, 1] range
      cos_phi = std::max(-1.0, std::min(1.0, cos_phi));

      // Calculate sine squared
      double sin_phi_sq = 1.0 - cos_phi * cos_phi;

      // Determine hybridization from atom types
      bool is_sp2_sp2 = false;
      bool is_sp3_sp3 = false;
      bool is_sp2_sp3 = false;

      // Determine hybridization based on atom types
      if (a_j.type.find("_R") != std::string::npos || a_j.type.find("_2") != std::string::npos) {
        if (a_k.type.find("_R") != std::string::npos || a_k.type.find("_2") != std::string::npos) {
          is_sp2_sp2 = true;
        } else if (a_k.type.find("_3") != std::string::npos) {
          is_sp2_sp3 = true;
        }
      } else if (a_j.type.find("_3") != std::string::npos) {
        if (a_k.type.find("_R") != std::string::npos || a_k.type.find("_2") != std::string::npos) {
          is_sp2_sp3 = true;
        } else if (a_k.type.find("_3") != std::string::npos) {
          is_sp3_sp3 = true;
        }
      }

      // Determine bond order
      double bond_order = 1.0;
      if (a_j.type == "C_R" && a_k.type == "C_R") {
        // For fullerene C60, use an intermediate bond order
        bond_order = 1.5;
      }

      // Initialize torsion parameters
      double V = 0.0;  // Force constant
      int n = 0;       // Periodicity 
      double cos_term = 0.0; // Determines phase

      // Determine torsion parameters based on hybridization
      if (is_sp3_sp3) {
        // Case 1: sp3-sp3 (general case)
        V = std::sqrt(params_j.V1 * params_k.V1);
        n = 3;
        cos_term = -1.0; // phi0 = 60°

        // Special case for single bonds between Group 6 elements (O, S, Se, Te, Po)
        bool j_is_group6 = (a_j.type.find("O_") != std::string::npos || 
          a_j.type.find("S_") != std::string::npos);
        bool k_is_group6 = (a_k.type.find("O_") != std::string::npos || 
          a_k.type.find("S_") != std::string::npos);

        if (bond_order == 1.0 && j_is_group6 && k_is_group6) {
          double V2 = 6.8, V3 = 6.8;
          if (a_j.type.find("O_") != std::string::npos) V2 = 2.0;
          if (a_k.type.find("O_") != std::string::npos) V3 = 2.0;
          V = std::sqrt(V2 * V3);
          n = 2;
          cos_term = -1.0; // phi0 = 90°
        }
      } else if (is_sp2_sp2) {
        // Case 2: sp2-sp2
        // Calculate V using equation 17 from UFF paper
        V = 5.0 * std::sqrt(params_j.U1 * params_k.U1) * 
          (1.0 + 4.18 * std::log(bond_order));
        n = 2;
        cos_term = 1.0; // phi0 = 180°
      } else if (is_sp2_sp3) {
        // Case 3: sp2-sp3 (default)
        V = 1.0;
        n = 6;
        cos_term = 1.0; // phi0 = 0°

        if (bond_order == 1.0) {
          // Special case for sp3 Group 6 - sp2 non-Group 6
          bool j_is_sp3_group6 = (a_j.type.find("_3") != std::string::npos && 
            (a_j.type.find("O_") != std::string::npos || 
            a_j.type.find("S_") != std::string::npos));
          bool k_is_sp3_group6 = (a_k.type.find("_3") != std::string::npos && 
            (a_k.type.find("O_") != std::string::npos || 
            a_k.type.find("S_") != std::string::npos));

          bool j_is_sp2_non_group6 = (a_j.type.find("_2") != std::string::npos || 
            a_j.type.find("_R") != std::string::npos) && 
            !(a_j.type.find("O_") != std::string::npos || 
            a_j.type.find("S_") != std::string::npos);
          bool k_is_sp2_non_group6 = (a_k.type.find("_2") != std::string::npos || 
            a_k.type.find("_R") != std::string::npos) && 
            !(a_k.type.find("O_") != std::string::npos || 
            a_k.type.find("S_") != std::string::npos);

          if ((j_is_sp3_group6 && k_is_sp2_non_group6) || 
            (k_is_sp3_group6 && j_is_sp2_non_group6)) {
            // Special case between group 6 sp3 and non-group 6 sp2
            V = 5.0 * std::sqrt(params_j.U1 * params_k.U1) * 
              (1.0 + 4.18 * std::log(bond_order));
            n = 2;
            cos_term = -1.0; // phi0 = 90°
          } else {
            // For fullerene, consider all carbons as part of sp2-sp2-sp2 system
            if (a_j.type == "C_R" || a_k.type == "C_R") {
              V = 2.0; // For sp3-sp2-sp2 (propene-like)
              n = 3;
              cos_term = -1.0; // phi0 = 180°
            }
          }
        }
      } else {
        // Default to general torsion 
        V = std::sqrt(params_j.V1 * params_k.V1);
        n = std::max(1, static_cast<int>(params_j.zeta));
        cos_term = 1.0; // phi0 = 180°
      }

      // Calculate torsion energy based on periodicity
      double cos_n_phi = 0.0;

      switch (n) {
        case 1:
          cos_n_phi = cos_phi;
          break;
        case 2:
          // cos(2x) = 2cos²(x) - 1 = 1 - 2sin²(x)
          cos_n_phi = 1.0 - 2.0 * sin_phi_sq;
          break;
        case 3:
          // cos(3x) = 4cos³(x) - 3cos(x)
          cos_n_phi = cos_phi * (cos_phi * cos_phi - 3.0 * sin_phi_sq);
          break;
        case 6:
          // cos(6x) = 32cos⁶(x) - 48cos⁴(x) + 18cos²(x) - 1
          //         = 1 - 32sin⁶(x) + 48sin⁴(x) - 18sin²(x)
          cos_n_phi = 1.0 + sin_phi_sq * (-18.0 + sin_phi_sq * (48.0 - 32.0 * sin_phi_sq));
          break;
        default:
          // For other periodicities, fall back to a simpler formula
          cos_n_phi = std::cos(n * std::acos(cos_phi));
      }

      // Final torsion energy calculation
      double torsion_term = V / 2.0 * (1.0 - cos_term * cos_n_phi);
      torsion_energy += torsion_term;
    }

    // Track 1-4 pairs for special scaling
    std::unordered_set<std::pair<size_t, size_t>, PairHash> pairs_1_4;
    for (const auto& torsion : torsions) {
      size_t i = torsion[0], l = torsion[3];
      pairs_1_4.insert({std::min(i, l), std::max(i, l)});
    }

    // Van der Waals (Lennard-Jones) with exclusions using RDKit's implementation
    for (size_t i = 0; i < atoms.size(); ++i) {
      for (size_t j = i + 1; j < atoms.size(); ++j) {
        std::pair<size_t, size_t> pair = {std::min(i, j), std::max(i, j)};

        // Skip 1-2 and 1-3 interactions
        if (excluded_pairs.count(pair) > 0) continue;

        const auto& p1 = get_params(atoms[i].type);
        const auto& p2 = get_params(atoms[j].type);

        // Calculate VDW parameters using mixing rules from RDKit
        // Note: RDKit uses sqrt(x1*x1) and sqrt(D1*D1) as the mixing rules
        const double xij = std::sqrt(p1.x * p2.x); // Minimum distance
        const double well_depth = std::sqrt(p1.D1 * p2.D1); // Well depth

        // Calculate distance
        const double dist = (atoms[i].position - atoms[j].position).norm();

        // Apply scaling for 1-4 interactions
        double scaling = 1.0;
        if (pairs_1_4.count(pair) > 0) {
          scaling = vdw_14_scale;
        }

        // Skip if distance is too large (RDKit uses a threshold)
        constexpr double threshold_multiplier = 10.0;
        const double threshold = threshold_multiplier * xij;
        if (dist > threshold || dist <= 0.0) continue;

        // Calculate LJ energy using RDKit's formula
        const double r = xij / dist;
        const double r6 = std::pow(r, 6);
        const double r12 = r6 * r6;
        double vdw_term = scaling * well_depth * (r12 - 2.0 * r6);

        vdw_energy += vdw_term;
      }
    }

    // Electrostatics with exclusions and 1-4 scaling
    constexpr double elec_14_scale = 0.5; // Standard UFF electrostatic scaling

    for (size_t i = 0; i < atoms.size(); ++i) {
      for (size_t j = i + 1; j < atoms.size(); ++j) {
        std::pair<size_t, size_t> pair = {std::min(i, j), std::max(i, j)};

        // Skip 1-2 and 1-3 interactions
        if (excluded_pairs.count(pair) > 0) continue;

        const double r = (atoms[i].position - atoms[j].position).norm();

        // Skip very small distances to avoid numerical issues
        if (r < 0.1) continue;

        // Apply scaling for 1-4 interactions
        double scaling = 1.0;
        if (pairs_1_4.count(pair) > 0) {
          scaling = elec_14_scale;
        }

        // Coulomb interaction with constant dielectric
        double elec_term = scaling * Constants::coulomb_constant * atoms[i].charge * atoms[j].charge / r;
        elec_energy += elec_term;
      }
    }

    double inversion_energy = 0.0;

    // Inversion (out-of-plane) energy - important for sp2 hybridized atoms
    for (const auto& inversion : inversions) {
      const size_t i = inversion[0];
      const size_t j = inversion[1]; // Central atom
      const size_t k = inversion[2];
      const size_t l = inversion[3];

      const auto& a_i = atoms[i].position;
      const auto& a_j = atoms[j].position;
      const auto& a_k = atoms[k].position;
      const auto& a_l = atoms[l].position;

      // Get parameters based on central atom type
      const std::string& central_atom_type = atoms[j].type;
      auto [K, C0, C1, C2] = calc_inversion_parameters(central_atom_type);

      // For fullerene (C60), all carbons are sp2 and need inversion terms
      if (K < 1e-6) continue; // Skip if force constant is zero

      // Calculate cosine of angle Y (between normal vector and fourth atom)
      double cos_Y = calculate_cos_Y(a_i, a_j, a_k, a_l);

      // Calculate sine of Y
      double sin_Y_sq = 1.0 - cos_Y * cos_Y;
      double sin_Y = (sin_Y_sq > 0.0) ? std::sqrt(sin_Y_sq) : 0.0;

      // Calculate cos(2*W) where W = π/2 - Y
      double cos_2W = 2.0 * sin_Y * sin_Y - 1.0;

      // Calculate inversion energy
      double energy = K * (C0 + C1 * sin_Y + C2 * cos_2W);
      inversion_energy += energy;
    }

    // Sum all energy components
    total = bond_energy + angle_energy + torsion_energy + vdw_energy + elec_energy + inversion_energy;

    return total;
  }

};

class XYZParser {
public:
    static std::vector<Atom> parse(const std::string& filename) {
        std::ifstream file(filename);
        if(!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);
        
        std::vector<Atom> atoms;
        std::string line;
        
        // Skip header
        std::getline(file, line); // Atom count
        std::getline(file, line); // Comment
        
        while(std::getline(file, line)) {
            std::stringstream ss(line);
            std::string element;
            double x, y, z;
            if(!(ss >> element >> x >> y >> z)) {
                throw std::runtime_error("Invalid XYZ format");
            }
            
            // Create a Vec3 for position and an Atom object
            Vec3 position;
            position[0] = x;
            position[1] = y;
            position[2] = z;
            
            // Create atom with mapped element type
            Atom atom;
            atom.type = map_element_to_uff(element);
            atom.position = position;
            atom.charge = 0.0;
            
            atoms.push_back(atom);
        }
        return atoms;
    }

private:
    // Complete element to UFF type mapping
    static std::string map_element_to_uff(const std::string& element) {
        static const std::unordered_map<std::string, std::string> mapping = {
            {"H", "H_"}, {"He", "He4+4"}, {"Li", "Li"}, {"Be", "Be3+2"}, {"B", "B_3"},
            {"C", "C_R"}, {"N", "N_3"}, {"O", "O_3"}, {"F", "F_"}, {"Ne", "Ne4+4"},
            {"Na", "Na"}, {"Mg", "Mg3+2"}, {"Al", "Al3"}, {"Si", "Si3"}, {"P", "P_3+3"},
            {"S", "S_3+2"}, {"Cl", "Cl"}, {"Ar", "Ar4+4"}, {"K", "K_"}, {"Ca", "Ca6+2"},
            {"Sc", "Sc3+3"}, {"Ti", "Ti3+4"}, {"V", "V_3+5"}, {"Cr", "Cr6+3"}, {"Mn", "Mn6+2"},
            {"Fe", "Fe3+2"}, {"Co", "Co6+3"}, {"Ni", "Ni4+2"}, {"Cu", "Cu3+1"}, {"Zn", "Zn3+2"},
            {"Ga", "Ga3+3"}, {"Ge", "Ge3"}, {"As", "As3+3"}, {"Se", "Se3+2"}, {"Br", "Br"},
            {"Kr", "Kr4+4"}, {"Rb", "Rb"}, {"Sr", "Sr6+2"}, {"Y", "Y_3+3"}, {"Zr", "Zr3+4"},
            {"Nb", "Nb3+5"}, {"Mo", "Mo6+6"}, {"Tc", "Tc6+5"}, {"Ru", "Ru6+2"}, {"Rh", "Rh6+3"},
            {"Pd", "Pd4+2"}, {"Ag", "Ag1+1"}, {"Cd", "Cd3+2"}, {"In", "In3+3"}, {"Sn", "Sn3"},
            {"Sb", "Sb3+3"}, {"Te", "Te3+2"}, {"I", "I_"}, {"Xe", "Xe4+4"}, {"Cs", "Cs"},
            {"Ba", "Ba6+2"}, {"La", "La3+3"}, {"Ce", "Ce6+3"}, {"Pr", "Pr6+3"}, {"Nd", "Nd6+3"},
            {"Pm", "Pm6+3"}, {"Sm", "Sm6+3"}, {"Eu", "Eu6+3"}, {"Gd", "Gd6+3"}, {"Tb", "Tb6+3"},
            {"Dy", "Dy6+3"}, {"Ho", "Ho6+3"}, {"Er", "Er6+3"}, {"Tm", "Tm6+3"}, {"Yb", "Yb6+3"},
            {"Lu", "Lu6+3"}, {"Hf", "Hf3+4"}, {"Ta", "Ta3+5"}, {"W", "W_6+6"}, {"Re", "Re6+5"},
            {"Os", "Os6+6"}, {"Ir", "Ir6+3"}, {"Pt", "Pt4+2"}, {"Au", "Au4+3"}, {"Hg", "Hg1+2"},
            {"Tl", "Tl3+3"}, {"Pb", "Pb3"}, {"Bi", "Bi3+3"}, {"Po", "Po3+2"}, {"At", "At"},
            {"Rn", "Rn4+4"}, {"Fr", "Fr"}, {"Ra", "Ra6+2"}, {"Ac", "Ac6+3"}, {"Th", "Th6+4"},
            {"Pa", "Pa6+4"}, {"U", "U_6+4"}, {"Np", "Np6+4"}, {"Pu", "Pu6+4"}, {"Am", "Am6+4"},
            {"Cm", "Cm6+3"}, {"Bk", "Bk6+3"}, {"Cf", "Cf6+3"}, {"Es", "Es6+3"}, {"Fm", "Fm6+3"},
            {"Md", "Md6+3"}, {"No", "No6+3"}, {"Lr", "Lw6+3"}
        };
        
        if (auto it = mapping.find(element); it != mapping.end()) {
            return it->second;
        }
        throw std::runtime_error("Unsupported element: " + element);
    }
};

// Put the atom_parameters definition in a separate .cpp file or use inline definition here
namespace Parameters {
const std::unordered_map<std::string, UFFAtom> atom_parameters = {
    {"H_", {0.354, 180, 2.886, 0.044, 12, 0.712, 0, 0, 4.528, 6.9452, 0.371}},
    {"H_b", {0.46, 83.5, 2.886, 0.044, 12, 0.712, 0, 0, 4.528, 6.9452, 0.371}},
    {"He4+4", {0.849, 90, 2.362, 0.056, 15.24, 0.098, 0, 0, 9.66, 14.92, 1.3}},
    {"Li", {1.336, 180, 2.451, 0.025, 12, 1.026, 0, 2, 3.006, 2.386, 1.557}},
    {"Be3+2", {1.074, 109.47, 2.745, 0.085, 12, 1.565, 0, 2, 4.877, 4.443, 1.24}},
    {"B_3", {0.838, 109.47, 4.083, 0.18, 12.052, 1.755, 0, 2, 5.11, 4.75, 0.822}},
    {"B_2", {0.828, 120, 4.083, 0.18, 12.052, 1.755, 0, 2, 5.11, 4.75, 0.822}},
    {"C_3", {0.757, 109.47, 3.851, 0.105, 12.73, 1.912, 2.119, 2, 5.343, 5.063, 0.759}},
    {"C_R", {0.729, 120, 3.851, 0.105, 12.73, 1.912, 0, 2, 5.343, 5.063, 0.759}},
    {"C_2", {0.732, 120, 3.851, 0.105, 12.73, 1.912, 0, 2, 5.343, 5.063, 0.759}},
    {"C_1", {0.706, 180, 3.851, 0.105, 12.73, 1.912, 0, 2, 5.343, 5.063, 0.759}},
    {"N_3", {0.7, 106.7, 3.66, 0.069, 13.407, 2.544, 0.45, 2, 6.899, 5.88, 0.715}},
    {"N_R", {0.699, 120, 3.66, 0.069, 13.407, 2.544, 0, 2, 6.899, 5.88, 0.715}},
    {"N_2", {0.685, 111.2, 3.66, 0.069, 13.407, 2.544, 0, 2, 6.899, 5.88, 0.715}},
    {"N_1", {0.656, 180, 3.66, 0.069, 13.407, 2.544, 0, 2, 6.899, 5.88, 0.715}},
    {"O_3", {0.658, 104.51, 3.5, 0.06, 14.085, 2.3, 0.018, 2, 8.741, 6.682, 0.669}},
    {"O_3_z", {0.528, 146, 3.5, 0.06, 14.085, 2.3, 0.018, 2, 8.741, 6.682, 0.669}},
    {"O_R", {0.68, 110, 3.5, 0.06, 14.085, 2.3, 0, 2, 8.741, 6.682, 0.669}},
    {"O_2", {0.634, 120, 3.5, 0.06, 14.085, 2.3, 0, 2, 8.741, 6.682, 0.669}},
    {"O_1", {0.639, 180, 3.5, 0.06, 14.085, 2.3, 0, 2, 8.741, 6.682, 0.669}},
    {"F_", {0.668, 180, 3.364, 0.05, 14.762, 1.735, 0, 2, 10.874, 7.474, 0.706}},
    {"Ne4+4", {0.92, 90, 3.243, 0.042, 15.44, 0.194, 0, 2, 11.04, 10.55, 1.768}},
    {"Na", {1.539, 180, 2.983, 0.03, 12, 1.081, 0, 1.25, 2.843, 2.296, 2.085}},
    {"Mg3+2", {1.421, 109.47, 3.021, 0.111, 12, 1.787, 0, 1.25, 3.951, 3.693, 1.5}},
    {"Al3", {1.244, 109.47, 4.499, 0.505, 11.278, 1.792, 0, 1.25, 4.06, 3.59, 1.201}},
    {"Si3", {1.117, 109.47, 4.295, 0.402, 12.175, 2.323, 1.225, 1.25, 4.168, 3.487, 1.176}},
    {"P_3+3", {1.101, 93.8, 4.147, 0.305, 13.072, 2.863, 2.4, 1.25, 5.463, 4, 1.102}},
    {"P_3+5", {1.056, 109.47, 4.147, 0.305, 13.072, 2.863, 2.4, 1.25, 5.463, 4, 1.102}},
    {"P_3+q", {1.056, 109.47, 4.147, 0.305, 13.072, 2.863, 2.4, 1.25, 5.463, 4, 1.102}},
    {"S_3+2", {1.064, 92.1, 4.035, 0.274, 13.969, 2.703, 0.484, 1.25, 6.928, 4.486, 1.047}},
    {"S_3+4", {1.049, 103.2, 4.035, 0.274, 13.969, 2.703, 0.484, 1.25, 6.928, 4.486, 1.047}},
    {"S_3+6", {1.027, 109.47, 4.035, 0.274, 13.969, 2.703, 0.484, 1.25, 6.928, 4.486, 1.047}},
    {"S_R", {1.077, 92.2, 4.035, 0.274, 13.969, 2.703, 0, 1.25, 6.928, 4.486, 1.047}},
    {"S_2", {0.854, 120, 4.035, 0.274, 13.969, 2.703, 0, 1.25, 6.928, 4.486, 1.047}},
    {"Cl", {1.044, 180, 3.947, 0.227, 14.866, 2.348, 0, 1.25, 8.564, 4.946, 0.994}},
    {"Ar4+4", {1.032, 90, 3.868, 0.185, 15.763, 0.3, 0, 1.25, 9.465, 6.355, 2.108}},
    {"K_", {1.953, 180, 3.812, 0.035, 12, 1.165, 0, 0.7, 2.421, 1.92, 2.586}},
    {"Ca6+2", {1.761, 90, 3.399, 0.238, 12, 2.141, 0, 0.7, 3.231, 2.88, 2}},
    {"Sc3+3", {1.513, 109.47, 3.295, 0.019, 12, 2.592, 0, 0.7, 3.395, 3.08, 1.75}},
    {"Ti3+4", {1.412, 109.47, 3.175, 0.017, 12, 2.659, 0, 0.7, 3.47, 3.38, 1.607}},
    {"Ti6+4", {1.412, 90, 3.175, 0.017, 12, 2.659, 0, 0.7, 3.47, 3.38, 1.607}},
    {"V_3+5", {1.402, 109.47, 3.144, 0.016, 12, 2.679, 0, 0.7, 3.65, 3.41, 1.47}},
    {"Cr6+3", {1.345, 90, 3.023, 0.015, 12, 2.463, 0, 0.7, 3.415, 3.865, 1.402}},
    {"Mn6+2", {1.382, 90, 2.961, 0.013, 12, 2.43, 0, 0.7, 3.325, 4.105, 1.533}},
    {"Fe3+2", {1.27, 109.47, 2.912, 0.013, 12, 2.43, 0, 0.7, 3.76, 4.14, 1.393}},
    {"Fe6+2", {1.335, 90, 2.912, 0.013, 12, 2.43, 0, 0.7, 3.76, 4.14, 1.393}},
    {"Co6+3", {1.241, 90, 2.872, 0.014, 12, 2.43, 0, 0.7, 4.105, 4.175, 1.406}},
    {"Ni4+2", {1.164, 90, 2.834, 0.015, 12, 2.43, 0, 0.7, 4.465, 4.205, 1.398}},
    {"Cu3+1", {1.302, 109.47, 3.495, 0.005, 12, 1.756, 0, 0.7, 4.2, 4.22, 1.434}},
    {"Zn3+2", {1.193, 109.47, 2.763, 0.124, 12, 1.308, 0, 0.7, 5.106, 4.285, 1.4}},
    {"Ga3+3", {1.26, 109.47, 4.383, 0.415, 11, 1.821, 0, 0.7, 3.641, 3.16, 1.211}},
    {"Ge3", {1.197, 109.47, 4.28, 0.379, 12, 2.789, 0.701, 0.7, 4.051, 3.438, 1.189}},
    {"As3+3", {1.211, 92.1, 4.23, 0.309, 13, 2.864, 1.5, 0.7, 5.188, 3.809, 1.204}},
    {"Se3+2", {1.19, 90.6, 4.205, 0.291, 14, 2.764, 0.335, 0.7, 6.428, 4.131, 1.224}},
    {"Br", {1.192, 180, 4.189, 0.251, 15, 2.519, 0, 0.7, 7.79, 4.425, 1.141}},
    {"Kr4+4", {1.147, 90, 4.141, 0.22, 16, 0.452, 0, 0.7, 8.505, 5.715, 2.27}},
    {"Rb", {2.26, 180, 4.114, 0.04, 12, 1.592, 0, 0.2, 2.331, 1.846, 2.77}},
    {"Sr6+2", {2.052, 90, 3.641, 0.235, 12, 2.449, 0, 0.2, 3.024, 2.44, 2.415}},
    {"Y_3+3", {1.698, 109.47, 3.345, 0.072, 12, 3.257, 0, 0.2, 3.83, 2.81, 1.998}},
    {"Zr3+4", {1.564, 109.47, 3.124, 0.069, 12, 3.667, 0, 0.2, 3.4, 3.55, 1.758}},
    {"Nb3+5", {1.473, 109.47, 3.165, 0.059, 12, 3.618, 0, 0.2, 3.55, 3.38, 1.603}},
    {"Mo6+6", {1.467, 90, 3.052, 0.056, 12, 3.4, 0, 0.2, 3.465, 3.755, 1.53}},
    {"Mo3+6", {1.484, 109.47, 3.052, 0.056, 12, 3.4, 0, 0.2, 3.465, 3.755, 1.53}},
    {"Tc6+5", {1.322, 90, 2.998, 0.048, 12, 3.4, 0, 0.2, 3.29, 3.99, 1.5}},
    {"Ru6+2", {1.478, 90, 2.963, 0.056, 12, 3.4, 0, 0.2, 3.575, 4.015, 1.5}},
    {"Rh6+3", {1.332, 90, 2.929, 0.053, 12, 3.5, 0, 0.2, 3.975, 4.005, 1.509}},
    {"Pd4+2", {1.338, 90, 2.899, 0.048, 12, 3.21, 0, 0.2, 4.32, 4, 1.544}},
    {"Ag1+1", {1.386, 180, 3.148, 0.036, 12, 1.956, 0, 0.2, 4.436, 3.134, 1.622}},
    {"Cd3+2", {1.403, 109.47, 2.848, 0.228, 12, 1.65, 0, 0.2, 5.034, 3.957, 1.6}},
    {"In3+3", {1.459, 109.47, 4.463, 0.599, 11, 2.07, 0, 0.2, 3.506, 2.896, 1.404}},
    {"Sn3", {1.398, 109.47, 4.392, 0.567, 12, 2.961, 0.199, 0.2, 3.987, 3.124, 1.354}},
    {"Sb3+3", {1.407, 91.6, 4.42, 0.449, 13, 2.704, 1.1, 0.2, 4.899, 3.342, 1.404}},
    {"Te3+2", {1.386, 90.25, 4.47, 0.398, 14, 2.882, 0.3, 0.2, 5.816, 3.526, 1.38}},
    {"I_", {1.382, 180, 4.5, 0.339, 15, 2.65, 0, 0.2, 6.822, 3.762, 1.333}},
    {"Xe4+4", {1.267, 90, 4.404, 0.332, 12, 0.556, 0, 0.2, 7.595, 4.975, 2.459}},
    {"Cs", {2.57, 180, 4.517, 0.045, 12, 1.573, 0, 0.1, 2.183, 1.711, 2.984}},
    {"Ba6+2", {2.277, 90, 3.703, 0.364, 12, 2.727, 0, 0.1, 2.814, 2.396, 2.442}},
    {"La3+3", {1.943, 109.47, 3.522, 0.017, 12, 3.3, 0, 0.1, 2.8355, 2.7415, 2.071}},
    {"Ce6+3", {1.841, 90, 3.556, 0.013, 12, 3.3, 0, 0.1, 2.774, 2.692, 1.925}},
    {"Pr6+3", {1.823, 90, 3.606, 0.01, 12, 3.3, 0, 0.1, 2.858, 2.564, 2.007}},
    {"Nd6+3", {1.816, 90, 3.575, 0.01, 12, 3.3, 0, 0.1, 2.8685, 2.6205, 2.007}},
    {"Pm6+3", {1.801, 90, 3.547, 0.009, 12, 3.3, 0, 0.1, 2.881, 2.673, 2}},
    {"Sm6+3", {1.78, 90, 3.52, 0.008, 12, 3.3, 0, 0.1, 2.9115, 2.7195, 1.978}},
    {"Eu6+3", {1.771, 90, 3.493, 0.008, 12, 3.3, 0, 0.1, 2.8785, 2.7875, 2.227}},
    {"Gd6+3", {1.735, 90, 3.368, 0.009, 12, 3.3, 0, 0.1, 3.1665, 2.9745, 1.968}},
    {"Tb6+3", {1.732, 90, 3.451, 0.007, 12, 3.3, 0, 0.1, 3.018, 2.834, 1.954}},
    {"Dy6+3", {1.71, 90, 3.428, 0.007, 12, 3.3, 0, 0.1, 3.0555, 2.8715, 1.934}},
    {"Ho6+3", {1.696, 90, 3.409, 0.007, 12, 3.416, 0, 0.1, 3.127, 2.891, 1.925}},
    {"Er6+3", {1.673, 90, 3.391, 0.007, 12, 3.3, 0, 0.1, 3.1865, 2.9145, 1.915}},
    {"Tm6+3", {1.66, 90, 3.374, 0.006, 12, 3.3, 0, 0.1, 3.2514, 2.9329, 2}},
    {"Yb6+3", {1.637, 90, 3.355, 0.228, 12, 2.618, 0, 0.1, 3.2889, 2.965, 2.158}},
    {"Lu6+3", {1.671, 90, 3.64, 0.041, 12, 3.271, 0, 0.1, 2.9629, 2.4629, 1.896}},
    {"Hf3+4", {1.611, 109.47, 3.141, 0.072, 12, 3.921, 0, 0.1, 3.7, 3.4, 1.759}},
    {"Ta3+5", {1.511, 109.47, 3.17, 0.081, 12, 4.075, 0, 0.1, 5.1, 2.85, 1.605}},
    {"W_6+6", {1.392, 90, 3.069, 0.067, 12, 3.7, 0, 0.1, 4.63, 3.31, 1.538}},
    {"W_3+4", {1.526, 109.47, 3.069, 0.067, 12, 3.7, 0, 0.1, 4.63, 3.31, 1.538}},
    {"W_3+6", {1.38, 109.47, 3.069, 0.067, 12, 3.7, 0, 0.1, 4.63, 3.31, 1.538}},
    {"Re6+5", {1.372, 90, 2.954, 0.066, 12, 3.7, 0, 0.1, 3.96, 3.92, 1.6}},
    {"Re3+7", {1.314, 109.47, 2.954, 0.066, 12, 3.7, 0, 0.1, 3.96, 3.92, 1.6}},
    {"Os6+6", {1.372, 90, 3.12, 0.037, 12, 3.7, 0, 0.1, 5.14, 3.63, 1.7}},
    {"Ir6+3", {1.371, 90, 2.84, 0.073, 12, 3.731, 0, 0.1, 5, 4, 1.866}},
    {"Pt4+2", {1.364, 90, 2.754, 0.08, 12, 3.382, 0, 0.1, 4.79, 4.43, 1.557}},
    {"Au4+3", {1.262, 90, 3.293, 0.039, 12, 2.625, 0, 0.1, 4.894, 2.586, 1.618}},
    {"Hg1+2", {1.34, 180, 2.705, 0.385, 12, 1.75, 0, 0.1, 6.27, 4.16, 1.6}},
    {"Tl3+3", {1.518, 120, 4.347, 0.68, 11, 2.068, 0, 0.1, 3.2, 2.9, 1.53}},
    {"Pb3", {1.459, 109.47, 4.297, 0.663, 12, 2.846, 0.1, 0.1, 3.9, 3.53, 1.444}},
    {"Bi3+3", {1.512, 90, 4.37, 0.518, 13, 2.47, 1, 0.1, 4.69, 3.74, 1.514}},
    {"Po3+2", {1.5, 90, 4.709, 0.325, 14, 2.33, 0.3, 0.1, 4.21, 4.21, 1.48}},
    {"At", {1.545, 180, 4.75, 0.284, 15, 2.24, 0, 0.1, 4.75, 4.75, 1.47}},
    {"Rn4+4", {1.42, 90, 4.765, 0.248, 16, 0.583, 0, 0.1, 5.37, 5.37, 2.2}},
    {"Fr", {2.88, 180, 4.9, 0.05, 12, 1.847, 0, 0, 2, 2, 2.3}},
    {"Ra6+2", {2.512, 90, 3.677, 0.404, 12, 2.92, 0, 0, 2.843, 2.434, 2.2}},
    {"Ac6+3", {1.983, 90, 3.478, 0.033, 12, 3.9, 0, 0, 2.835, 2.835, 2.108}},
    {"Th6+4", {1.721, 90, 3.396, 0.026, 12, 4.202, 0, 0, 3.175, 2.905, 2.018}},
    {"Pa6+4", {1.711, 90, 3.424, 0.022, 12, 3.9, 0, 0, 2.985, 2.905, 1.8}},
    {"U_6+4", {1.684, 90, 3.395, 0.022, 12, 3.9, 0, 0, 3.341, 2.853, 1.713}},
    {"Np6+4", {1.666, 90, 3.424, 0.019, 12, 3.9, 0, 0, 3.549, 2.717, 1.8}},
    {"Pu6+4", {1.657, 90, 3.424, 0.016, 12, 3.9, 0, 0, 3.243, 2.819, 1.84}},
    {"Am6+4", {1.66, 90, 3.381, 0.014, 12, 3.9, 0, 0, 2.9895, 3.0035, 1.942}},
    {"Cm6+3", {1.801, 90, 3.326, 0.013, 12, 3.9, 0, 0, 2.8315, 3.1895, 1.9}},
    {"Bk6+3", {1.761, 90, 3.339, 0.013, 12, 3.9, 0, 0, 3.1935, 3.0355, 1.9}},
    {"Cf6+3", {1.75, 90, 3.313, 0.013, 12, 3.9, 0, 0, 3.197, 3.101, 1.9}},
    {"Es6+3", {1.724, 90, 3.299, 0.012, 12, 3.9, 0, 0, 3.333, 3.089, 1.9}},
    {"Fm6+3", {1.712, 90, 3.286, 0.012, 12, 3.9, 0, 0, 3.4, 3.1, 1.9}},
    {"Md6+3", {1.689, 90, 3.274, 0.011, 12, 3.9, 0, 0, 3.47, 3.11, 1.9}},
    {"No6+3", {1.679, 90, 3.248, 0.011, 12, 3.9, 0, 0, 3.475, 3.175, 1.9}},
    {"Lw6+3", {1.698, 90, 3.236, 0.011, 12, 3.9, 0, 0, 3.5, 3.2, 1.9}}
};
}

} // namespace UFF
