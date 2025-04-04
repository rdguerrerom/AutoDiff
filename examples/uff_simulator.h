#pragma once

#include "uff_parameters.h"
#include <vector>
#include <cmath>
#include <unordered_map>
#include <algorithm>

// 3D vector class
struct Vec3 {
    double x, y, z;
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    double norm() const { return std::sqrt(x*x + y*y + z*z); }
    Vec3 operator-(const Vec3& other) const { return Vec3(x - other.x, y - other.y, z - other.z); }
    double dot(const Vec3& other) const { return x*other.x + y*other.y + z*other.z; }
    Vec3 cross(const Vec3& other) const {
        return Vec3(y*other.z - z*other.y, z*other.x - x*other.z, x*other.y - y*other.x);
    }
};

// Atom definition
struct Atom {
    std::string type;  // Atom type (e.g., "C_sp2", "Ar")
    Vec3 position;     // Coordinates (Å)
    Atom(const std::string& type, const Vec3& pos) : type(type), position(pos) {}
};

// Energy terms
struct UFFBond { int a1, a2; double r0, D_e, alpha; };
struct UFFAngle { int a1, a2, a3; double theta0, k_theta; };
struct UFFTorsion { int a1, a2, a3, a4; double V_n, phi0; int n; };
struct UFFInversion { int a1, a2, a3, a4; double chi0, k_inv; };

class UFFSimulator {
private:
    std::vector<UFFBond> bonds;
    std::vector<UFFAngle> angles;
    std::vector<UFFTorsion> torsions;
    std::vector<UFFInversion> inversions;
    double coulomb_const = 332.0637; // kcal·mol⁻¹·Å·e⁻²

    // Calculate angle between three points (radians)
    double computeAngle(const Vec3& a, const Vec3& b, const Vec3& c) {
        Vec3 ba = a - b, bc = c - b;
        return std::acos(ba.dot(bc) / (ba.norm() * bc.norm()));
    }

    // Calculate dihedral angle (radians)
    double computeDihedral(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
        Vec3 ab = b - a, bc = c - b, cd = d - c;
        Vec3 n1 = ab.cross(bc), n2 = bc.cross(cd);
        return std::atan2(n1.cross(n2).dot(bc) / bc.norm(), n1.dot(n2));
    }

public:
    // Add topology terms
    void addBond(const UFFBond& bond) { bonds.push_back(bond); }
    void addAngle(const UFFAngle& angle) { angles.push_back(angle); }
    void addTorsion(const UFFTorsion& torsion) { torsions.push_back(torsion); }
    void addInversion(const UFFInversion& inv) { inversions.push_back(inv); }

    // Compute total energy
    double computeEnergy(const std::vector<Atom>& atoms) {
        double energy = 0.0;

        // 1. Bond stretching (Morse potential)
        for (const auto& bond : bonds) {
            const Vec3& pos1 = atoms[bond.a1].position;
            const Vec3& pos2 = atoms[bond.a2].position;
            double r = (pos2 - pos1).norm();
            double exp_term = std::exp(-bond.alpha * (r - bond.r0));
            energy += bond.D_e * (1 - exp_term) * (1 - exp_term) - bond.D_e;
        }

        // 2. Angle bending (harmonic cosine)
        for (const auto& angle : angles) {
            const Vec3& pos1 = atoms[angle.a1].position;
            const Vec3& pos2 = atoms[angle.a2].position;
            const Vec3& pos3 = atoms[angle.a3].position;
            double theta = computeAngle(pos1, pos2, pos3);
            energy += angle.k_theta * std::pow(std::cos(theta) - std::cos(angle.theta0), 2);
        }

        // 3. Torsional interactions
        for (const auto& torsion : torsions) {
            const Vec3& pos1 = atoms[torsion.a1].position;
            const Vec3& pos2 = atoms[torsion.a2].position;
            const Vec3& pos3 = atoms[torsion.a3].position;
            const Vec3& pos4 = atoms[torsion.a4].position;
            double phi = computeDihedral(pos1, pos2, pos3, pos4);
            energy += torsion.V_n / 2 * (1 - std::cos(torsion.n * phi - torsion.phi0));
        }

        // 4. Inversion terms
        for (const auto& inv : inversions) {
            const Vec3& pos1 = atoms[inv.a1].position;
            const Vec3& pos2 = atoms[inv.a2].position;
            const Vec3& pos3 = atoms[inv.a3].position;
            const Vec3& pos4 = atoms[inv.a4].position;
            Vec3 normal = (pos3 - pos2).cross(pos4 - pos2);
            double sin_chi = (pos1 - pos2).dot(normal) / ((pos1 - pos2).norm() * normal.norm());
            sin_chi = std::clamp(sin_chi, -1.0, 1.0);
            double chi = std::asin(sin_chi);
            energy += inv.k_inv * std::pow(chi - inv.chi0, 2);
        }

        // 5. Van der Waals (Lennard-Jones)
        for (size_t i = 0; i < atoms.size(); ++i) {
            for (size_t j = i + 1; j < atoms.size(); ++j) {
                double r = (atoms[i].position - atoms[j].position).norm();
                const auto& type1 = atoms[i].type;
                const auto& type2 = atoms[j].type;

                // Check for custom C-X pairs
                double sigma, epsilon;
                auto it1 = custom_cx_pairs.find(type1);
                if (it1 != custom_cx_pairs.end()) {
                    auto it2 = it1->second.find(type2);
                    if (it2 != it1->second.end()) {
                        sigma = it2->second.first;
                        epsilon = it2->second.second;
                        goto compute_lj; // Skip mixing rules
                    }
                }
                // Standard mixing rules
                sigma = (uff_atom_params[type1]["sigma"] + uff_atom_params[type2]["sigma"]) / 2;
                epsilon = std::sqrt(uff_atom_params[type1]["epsilon"] * uff_atom_params[type2]["epsilon"]);

                compute_lj:
                energy += 4 * epsilon * (std::pow(sigma / r, 12) - std::pow(sigma / r, 6));
            }
        }

        // 6. Electrostatic (Coulomb)
        for (size_t i = 0; i < atoms.size(); ++i) {
            for (size_t j = i + 1; j < atoms.size(); ++j) {
                double r = (atoms[i].position - atoms[j].position).norm();
                double q1 = uff_atom_params[atoms[i].type]["charge"];
                double q2 = uff_atom_params[atoms[j].type]["charge"];
                energy += coulomb_const * q1 * q2 / r;
            }
        }

        return energy;
    }
};

