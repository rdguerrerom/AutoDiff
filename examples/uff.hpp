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
#include <set>
#include <iostream>
#include <tuple>
#include <utility>

namespace UFF {

/**
 * @brief Enumeration for atom hybridization types
 */
enum HybridizationType {
  UNSPECIFIED = 0,
  S,
  SP,
  SP2,
  SP3,
  SP2D,
  SP3D,
  SP3D2,
  OTHER
};

/**
 * @brief Constants used throughout the UFF implementation
 * RDKIT REF: ForceFields::UFF::Params namespace constants
 */
namespace Constants {
constexpr double kcal_to_eV = 0.0433641;
constexpr double coulomb_constant = 332.0637;
constexpr double pi = 3.14159265358979323846;
constexpr double deg2rad = pi / 180.0;
}

/**
 * @brief 3D vector structure with common vector operations
 * RDKIT REF: RDGeom::Point3D in RDKit
 */
struct Vec3 : std::array<double, 3> {
  using Base = std::array<double, 3>;
  using Base::Base;

  /**
   * @brief Calculate the Euclidean norm (length) of the vector
   * @return The length of the vector
   */
  double norm() const noexcept {
    return std::sqrt((*this)[0]*(*this)[0] + (*this)[1]*(*this)[1] + (*this)[2]*(*this)[2]);
  }

  /**
   * @brief Vector subtraction operator
   * @param other Vector to subtract
   * @return Result of subtraction
   */
  Vec3 operator-(const Vec3& other) const noexcept {
    Vec3 result;
    result[0] = (*this)[0] - other[0];
    result[1] = (*this)[1] - other[1];
    result[2] = (*this)[2] - other[2];
    return result;
  }

  /**
   * @brief Scalar multiplication operator
   * @param scalar Value to multiply by
   * @return Result of multiplication
   */
  Vec3 operator*(double scalar) const noexcept {
    Vec3 result;
    result[0] = (*this)[0] * scalar;
    result[1] = (*this)[1] * scalar;
    result[2] = (*this)[2] * scalar;
    return result;
  }

  /**
   * @brief Dot product operation
   * @param other Vector to calculate dot product with
   * @return Scalar dot product result
   */
  double dot(const Vec3& other) const noexcept {
    return (*this)[0]*other[0] + (*this)[1]*other[1] + (*this)[2]*other[2];
  }

  /**
   * @brief Cross product operation
   * @param other Vector to calculate cross product with
   * @return Vector cross product result
   */
  Vec3 cross(const Vec3& other) const noexcept {
    Vec3 result;
    result[0] = (*this)[1]*other[2] - (*this)[2]*other[1];
    result[1] = (*this)[2]*other[0] - (*this)[0]*other[2];
    result[2] = (*this)[0]*other[1] - (*this)[1]*other[0];
    return result;
  }
  
  /**
   * @brief Create a normalized vector with the same direction
   */
  void normalize() {
    double len = norm();
    if (len > 1e-8) {
      (*this)[0] /= len;
      (*this)[1] /= len;
      (*this)[2] /= len;
    }
  }
};

/**
 * @brief Non-member scalar multiplication operator
 * @param scalar Value to multiply by
 * @param v Vector to multiply
 * @return Result of multiplication
 */
inline Vec3 operator*(double scalar, const Vec3& v) noexcept {
  return v * scalar;
}

/**
 * @brief Atom structure containing type, position, and charge information
 * RDKIT REF: Similar to RDKit's atom representation
 */
struct Atom {
  std::string type;
  Vec3 position;
  double charge;
};

/**
 * @brief Energy components container for tracking individual energy contributions
 * RDKIT REF: ForceFields::ForceField::calcEnergy() which calculates these components
 */
struct EnergyComponents {
  double bond_energy = 0.0;
  double angle_energy = 0.0;
  double torsion_energy = 0.0;
  double vdw_energy = 0.0;
  double elec_energy = 0.0;
  double inversion_energy = 0.0;

  /**
   * @brief Calculate total energy from all components
   * @return Sum of all energy components
   */
  double total() const noexcept {
    return bond_energy + angle_energy + torsion_energy +
           vdw_energy + elec_energy + inversion_energy;
  }

  /**
   * @brief Print energy components for debugging
   */
  void print() const {
    std::cout << "Energy components (kcal/mol):" << std::endl;
    std::cout << "  Bond energy: " << bond_energy << std::endl;
    std::cout << "  Angle energy: " << angle_energy << std::endl;
    std::cout << "  Torsion energy: " << torsion_energy << std::endl;
    std::cout << "  Inversion energy: " << inversion_energy << std::endl;
    std::cout << "  VdW energy: " << vdw_energy << std::endl;
    std::cout << "  Electrostatic energy: " << elec_energy << std::endl;
  }
};

namespace Parameters {
/**
 * @brief UFF atom parameters for energy calculations
 * RDKIT REF: ForceFields::UFF::AtomicParams
 */
struct UFFAtom {
  double r1;          ///< Valence bond radius
  double theta0_deg;  ///< Valence angle in degrees
  double x;           ///< vdW characteristic length
  double D1;          ///< vdW atomic energy
  double zeta;        ///< vdW scaling term
  double Z1;          ///< Effective charge
  double V1;          ///< sp3 torsional barrier parameter
  double U1;          ///< Torsional contribution for sp2-sp3 bonds
  double GMP_Xi;      ///< GMP Electronegativity
  double GMP_Hardness;///< GMP Hardness
  double GMP_Radius;  ///< GMP Radius value
  HybridizationType hybrid;  ///< Hybridization type

  /**
   * @brief Get valence angle in radians
   * @return Valence angle in radians
   */
  double theta0() const noexcept { return theta0_deg * Constants::deg2rad; }
};

// Atom parameters map
extern const std::unordered_map<std::string, UFFAtom> atom_parameters;

        // Declaration of the electronegativity map
        extern const std::unordered_map<std::string, double> electronegativity;
        
        // Function to get electronegativity with fallback
        double get_electronegativity(const std::string& uff_type);
}

/**
 * @brief Custom hash function for std::pair
 */
struct PairHash {
  /**
   * @brief Hash function operator for std::pair
   * @tparam T1 First element type
   * @tparam T2 Second element type
   * @param pair Pair to hash
   * @return Hash value
   */
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    std::size_t h1 = std::hash<T1>{}(pair.first);
    std::size_t h2 = std::hash<T2>{}(pair.second);
    return h1 ^ (h2 << 1); // Simple hash combine function
  }
};

/**
 * @brief Utility class for building molecular topology
 * RDKIT REF: RDKit's topology building in ForceFields
 */
class TopologyBuilder {
public:
  /**
   * @brief Detect bonds in a molecular structure
   * @param atoms Vector of atoms
   * @param tolerance Tolerance factor for bond detection
   * @return Vector of bond pairs (indices)
   */
  static std::vector<std::array<size_t, 2>> detect_bonds(
    const std::vector<Atom>& atoms, 
    double tolerance = 1.1  
  );

  /**
   * @brief Detect angles based on bond connectivity
   * @param bonds Vector of bonds
   * @return Vector of angle triplets (indices)
   */
  static std::vector<std::array<size_t, 3>> detect_angles(
    const std::vector<std::array<size_t, 2>>& bonds
  );

  /**
   * @brief Detect torsion (dihedral) angles based on bond connectivity
   * @param bonds Vector of bonds
   * @return Vector of torsion quadruplets (indices)
   */
  static std::vector<std::array<size_t, 4>> detect_torsions(
    const std::vector<std::array<size_t, 2>>& bonds
  );

  /**
   * @brief Detect inversion (improper) terms based on atoms and bonds
   * @param atoms Vector of atoms
   * @param bonds Vector of bonds
   * @return Vector of inversion quadruplets (indices)
   */
  static std::vector<std::array<size_t, 4>> detect_inversions(
    const std::vector<Atom>& atoms,
    const std::vector<std::array<size_t, 2>>& bonds
  );
};

/**
 * @brief Main class for UFF energy calculations
 * RDKIT REF: ForceFields::ForceField and ForceFields::UFF builders
 */
class EnergyCalculator {
private:
  std::vector<Atom> atoms;
  std::vector<std::array<size_t, 2>> bonds;
  std::vector<std::array<size_t, 3>> angles;
  std::vector<std::array<size_t, 4>> torsions;
  std::vector<std::array<size_t, 4>> inversions;
  std::unordered_set<std::pair<size_t, size_t>, PairHash> excluded_pairs;

  // Scaling factors for 1-4 interactions - matches RDKit values
  static constexpr double vdw_14_scale = 0.5;
  static constexpr double elec_14_scale = 0.75;

  /**
   * @brief Get UFF atom parameters for a given atom type
   * @param type Atom type string
   * @return Reference to atom parameters
   * @throws std::runtime_error if parameters not found
   */
  const Parameters::UFFAtom& get_params(const std::string& type) const;

  /**
   * @brief Calculate bond order based on atom types
   * @param type_i First atom type
   * @param type_j Second atom type
   * @return Bond order value
   */
  double calculate_bond_order(const std::string& type_i, const std::string& type_j) const;

  /**
   * @brief Calculate natural bond length with corrections
   * @param type_i First atom type
   * @param type_j Second atom type
   * @param bond_order Bond order
   * @return Natural bond length
   */
  double calculate_natural_bond_length(
      const std::string& type_i, 
      const std::string& type_j, 
      double bond_order) const;

  /**
   * @brief Calculate bond force constant
   * @param type_i First atom type
   * @param type_j Second atom type
   * @param rest_length Natural bond length
   * @return Force constant
   */
  double calculate_bond_force_constant(
      const std::string& type_i,
      const std::string& type_j,
      double rest_length) const;

  /**
   * @brief Calculate bond stretching energy
   * @return Bond energy contribution
   */
  double calculate_bond_energy() const;

  /**
   * @brief Calculate angle bending energy
   * @return Angle energy contribution
   */
  double calculate_angle_energy() const;

  /**
   * @brief Calculate torsional energy
   * @return Torsion energy contribution
   */
  double calculate_torsion_energy() const;

  /**
   * @brief Calculate inversion energy
   * @return Inversion energy contribution
   */
  double calculate_inversion_energy() const;

  /**
   * @brief Calculate van der Waals (Lennard-Jones) energy
   * @return VdW energy contribution
   */
  double calculate_vdw_energy() const;

  /**
   * @brief Calculate electrostatic energy
   * @return Electrostatic energy contribution
   */
  double calculate_electrostatic_energy() const;

  /**
   * @brief Determine if atom belongs to Group 6 elements
   * @param atom_type Atom type string
   * @return True if atom is in Group 6, false otherwise
   */
  bool is_group6(const std::string& atom_type) const;

  /**
   * @brief Get all 1-4 atom pairs (separated by 3 bonds)
   * @return Set of 1-4 atom index pairs
   */
  std::unordered_set<std::pair<size_t, size_t>, PairHash> get_1_4_pairs() const;

  /**
   * @brief Calculate inversion parameters based on central atom type
   * @param central_atom_type Type of central atom
   * @return Tuple of K, C0, C1, C2 parameters
   */
  static std::tuple<double, double, double, double> calc_inversion_parameters(
      const std::string& central_atom_type);

  /**
   * @brief Calculate cosine of the out-of-plane angle
   * @param p1 First atom position
   * @param p2 Central atom position
   * @param p3 Third atom position
   * @param p4 Fourth atom position
   * @return Cosine of the out-of-plane angle
   */
  static double calculate_cos_Y(
      const Vec3& p1, const Vec3& p2, const Vec3& p3, const Vec3& p4);
      
  /**
   * @brief Assign partial charges to atoms for electrostatic calculations
   * Uses a simple electronegativity-based method
   */
  void assign_partial_charges();

public:
  /**
   * @brief Constructor for EnergyCalculator
   * @param atoms Vector of atoms
   * @param bonds Vector of bonds
   * @param angles Vector of angles
   * @param torsions Vector of torsions
   * @param inversions Vector of inversions (optional, can be auto-detected)
   */
  EnergyCalculator(std::vector<Atom> atoms,
                   std::vector<std::array<size_t, 2>> bonds,
                   std::vector<std::array<size_t, 3>> angles,
                   std::vector<std::array<size_t, 4>> torsions,
                   std::vector<std::array<size_t, 4>> inversions = {});

  /**
   * @brief Calculate the total UFF energy
   * @return Total energy value
   */
  double calculate() const;
};

/**
 * @brief XYZ file parser for molecular structures
 * RDKIT REF: Based on RDKit's file reading functionality
 */
class XYZParser {
public:
  /**
   * @brief Parse an XYZ file into UFF atoms
   * @param filename XYZ file path
   * @return Vector of atoms with positions and types
   * @throws std::runtime_error if file cannot be opened or format is invalid
   */
  static std::vector<Atom> parse(const std::string& filename);

private:
  /**
   * @brief Process a single atom line from an XYZ file
   * @param line The line to process
   * @param atoms Vector to add the parsed atom to
   * @param lineNum Line number for error reporting
   * @param is_tinker_format Whether the file is in Tinker XYZ format
   */
  static void processAtomLine(const std::string& line, std::vector<Atom>& atoms, 
                             int lineNum, bool is_tinker_format);

  /**
   * @brief Normalize element symbol to standard form
   * @param raw_symbol Raw element symbol from XYZ file
   * @return Normalized element symbol
   */
  static std::string normalize_element_symbol(const std::string& raw_symbol);

  /**
   * @brief Map element symbols to UFF atom types
   * @param element Element symbol
   * @return UFF atom type string
   * @throws std::runtime_error for unsupported elements
   */
  static std::string map_element_to_uff(const std::string& element);
  
  /**
   * @brief Refine atom types based on molecular geometry and connectivity
   * @param atoms Vector of atoms with positions
   */
  static void refine_atom_types(std::vector<Atom>& atoms);
  
  /**
   * @brief Get expected bond length between two atom types
   * @param type1 First atom type
   * @param type2 Second atom type
   * @return Expected bond length in Angstroms
   */
  static double get_expected_bond_length(const std::string& type1, const std::string& type2);
  
  /**
   * @brief Calculate distance between two atoms, handling periodic boundary conditions
   * @param pos1 Position of first atom
   * @param pos2 Position of second atom
   * @param cell_dimensions Optional cell dimensions for periodic systems
   * @return Distance in Angstroms
   */
  static double calculate_distance(const Vec3& pos1, const Vec3& pos2, 
                                  const Vec3* cell_dimensions = nullptr);
  
  /**
   * @brief Calculate bond angle between three atoms
   * @param pos1 Position of first atom
   * @param pos2 Position of central atom
   * @param pos3 Position of third atom
   * @param cell_dimensions Optional cell dimensions for periodic systems
   * @return Angle in radians
   */
  static double calculate_angle(const Vec3& pos1, const Vec3& pos2, const Vec3& pos3, 
                               const Vec3* cell_dimensions = nullptr);
};

} // namespace UFF
