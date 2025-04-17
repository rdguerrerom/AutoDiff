#include "uff.hpp"
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>
#include <regex>
#include <iomanip>

namespace UFF {

// Parse XYZ file into UFF atoms
std::vector<Atom> XYZParser::parse(const std::string& filename) {
    std::ifstream file(filename);
    if(!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::vector<Atom> atoms;
    std::string line;

    // Read number of atoms
    std::getline(file, line);
    int numAtoms = 0;
    try {
        // Strip any whitespace and convert to integer
        line = std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");
        numAtoms = std::stoi(line);
        if (numAtoms <= 0) {
            throw std::runtime_error("Invalid atom count: must be positive");
        }
    } catch(const std::exception& e) {
        throw std::runtime_error("Invalid XYZ format: could not parse atom count - " + std::string(e.what()));
    }

    // Read comment/title line
    std::getline(file, line);
    
    // Prepare container for atoms
    atoms.reserve(numAtoms);
    int lineNum = 2; // Already read 2 lines
    
    // Read first data line to determine format (standard XYZ or Tinker XYZ)
    bool is_tinker_format = false;
    std::string first_atom_line;
    
    if (std::getline(file, first_atom_line)) {
        lineNum++;
        
        // Skip empty lines and comments
        if (first_atom_line.empty() || first_atom_line[0] == '#' || first_atom_line[0] == '!') {
            // If the first line is a comment, process next lines normally
        } else {
            // Check if the first atom line is in Tinker format by examining the parts
            std::vector<std::string> parts;
            std::istringstream iss(std::regex_replace(first_atom_line, std::regex("^\\s+|\\s+$"), ""));
            std::string part;
            while (iss >> part) {
                parts.push_back(part);
            }
            
            // If we have enough parts for a Tinker format line
            if (parts.size() >= 5) {
                try {
                    int seq_num = std::stoi(parts[0]);
                    // Check if second part is a valid element (starts with letter)
                    if (std::isalpha(parts[1][0])) {
                        is_tinker_format = true;
                    }
                }
                catch (const std::exception&) {
                    // If we can't parse the first part as an integer, it's probably standard XYZ
                    is_tinker_format = false;
                }
            }
            
            // Process the first line that we just read
            processAtomLine(first_atom_line, atoms, lineNum, is_tinker_format);
        }
    }
    
    // Read remaining atom coordinates
    while(std::getline(file, line) && atoms.size() < static_cast<size_t>(numAtoms)) {
        lineNum++;
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line[0] == '!') {
            continue;
        }
        
        processAtomLine(line, atoms, lineNum, is_tinker_format);
    }

    // Validate atom count
    if (atoms.size() != static_cast<size_t>(numAtoms)) {
        throw std::runtime_error("XYZ file contained " + std::to_string(atoms.size()) + 
                               " atoms but header specified " + std::to_string(numAtoms));
    }
    
    // Update atom types based on molecular structure
    refine_atom_types(atoms);
    
    return atoms;
}

// Helper method to process a single atom line
void XYZParser::processAtomLine(const std::string& line, std::vector<Atom>& atoms, int lineNum, bool is_tinker_format) {
    // Trim the line and split by whitespace
    std::string trimmed_line = std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");
    std::vector<std::string> parts;
    std::istringstream iss(trimmed_line);
    std::string part;
    while (iss >> part) {
        parts.push_back(part);
    }
    
    // Check if we have enough data
    if (is_tinker_format) {
        // Tinker format: SeqNum Element X Y Z [AtomType] [ConnectedAtoms]
        if (parts.size() < 5) {
            throw std::runtime_error("Invalid Tinker XYZ format at line " + std::to_string(lineNum) + 
                                    ": insufficient data, expected 'SeqNum Element X Y Z [AtomType] [ConnectedAtoms]'");
        }
    } else {
        // Standard XYZ format: Element X Y Z
        if (parts.size() < 4) {
            throw std::runtime_error("Invalid standard XYZ format at line " + std::to_string(lineNum) + 
                                    ": insufficient data, expected 'Element X Y Z'");
        }
    }
    
    std::string element;
    double x, y, z;
    int atom_type = 0;  // Default atom type
    
    try {
        if (is_tinker_format) {
            // Parse Tinker format
            // int seq_num = std::stoi(parts[0]);  // We don't need to use this value
            element = parts[1];
            x = std::stod(parts[2]);
            y = std::stod(parts[3]);
            z = std::stod(parts[4]);
            
            // Try to read atom type if present
            if (parts.size() > 5) {
                atom_type = std::stoi(parts[5]);
            }
            
            // Connected atoms would be in parts[6] and beyond, but we'll detect bonds later
        } else {
            // Parse standard XYZ format
            element = parts[0];
            x = std::stod(parts[1]);
            y = std::stod(parts[2]);
            z = std::stod(parts[3]);
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Invalid XYZ format at line " + std::to_string(lineNum) + 
                                ": could not parse data - " + std::string(e.what()));
    }
    
    // Create and add the atom to our collection
    Vec3 position;
    position[0] = x;
    position[1] = y;
    position[2] = z;

    // Normalize element symbol
    std::string normalizedElement = normalize_element_symbol(element);
    
    Atom atom;
    atom.type = map_element_to_uff(normalizedElement);
    atom.position = position;
    atom.charge = 0.0;  // Default to zero charge

    atoms.push_back(atom);
}

// Normalize element symbol to standard form (e.g., "c" -> "C", "NA" -> "Na")
std::string XYZParser::normalize_element_symbol(const std::string& raw_symbol) {
    // Handle common cases first
    if (raw_symbol.empty()) {
        throw std::runtime_error("Empty atom symbol encountered");
    }
    
    // Check if the element symbol already includes UFF typing (e.g., "C_3", "N_R")
    if (raw_symbol.find('_') != std::string::npos) {
        return raw_symbol; // Already in UFF format, return as is
    }
    
    // Extract element symbol from potential format with numbers (e.g. "C1", "H2")
    std::string element;
    
    // First, check if there are any alphabetic characters
    bool has_alpha = false;
    for (char c : raw_symbol) {
        if (std::isalpha(c)) {
            has_alpha = true;
            break;
        }
    }
    
    if (!has_alpha) {
        throw std::runtime_error("Invalid atom symbol (no alphabetic characters): " + raw_symbol);
    }
    
    // Extract alphabetic part of the symbol
    for (char c : raw_symbol) {
        if (std::isalpha(c)) {
            element.push_back(c);
        } else if (!element.empty()) {
            // Stop after first numeric part following alphabetic characters
            break;
        }
    }
    
    // If still empty after processing, throw error
    if (element.empty()) {
        throw std::runtime_error("Invalid atom symbol: " + raw_symbol);
    }
    
    // Convert to proper capitalization: first letter uppercase, rest lowercase
    std::string normalized;
    normalized.push_back(std::toupper(element[0]));
    for (size_t i = 1; i < element.size(); ++i) {
        normalized.push_back(std::tolower(element[i]));
    }
    
    // Validate against known elements (optional)
    static const std::set<std::string> valid_elements = {
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", 
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", 
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", 
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", 
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", 
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", 
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", 
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", 
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", 
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", 
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    };
    
    if (valid_elements.find(normalized) == valid_elements.end()) {
        // Unknown element, but we'll attempt to process it anyway
        std::cerr << "Warning: Unrecognized element symbol '" << normalized 
                  << "' - attempting to proceed" << std::endl;
    }
    
    return normalized;
}

// Map element symbols to UFF atom types
std::string XYZParser::map_element_to_uff(const std::string& element) {
    // Check if it's already in UFF format
    if (element.find('_') != std::string::npos) {
        return element;
    }
    
    // Standard mapping of elements to their common UFF atom types
    static const std::unordered_map<std::string, std::string> mapping = {
        {"H", "H_"}, {"He", "He4+4"}, {"Li", "Li"}, {"Be", "Be3+2"}, {"B", "B_3"},
        {"C", "C_3"}, {"N", "N_3"}, {"O", "O_3"}, {"F", "F_"}, {"Ne", "Ne4+4"},
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
        {"Md", "Md6+3"}, {"No", "No6+3"}, {"Lw", "Lw6+3"}
    };

    if (auto it = mapping.find(element); it != mapping.end()) {
        return it->second;
    }
    
    // Element not found in mapping
    throw std::runtime_error("Unsupported element: " + element);
}

// Refine atom types based on molecular geometry and connectivity
void XYZParser::refine_atom_types(std::vector<Atom>& atoms) {
    // Skip if empty
    if (atoms.empty()) return;
    
    // Build preliminary connectivity based on interatomic distances
    std::vector<std::vector<size_t>> connectivity(atoms.size());
    std::vector<double> bond_distances;
    
    // Check if this might be C60 fullerene
    bool might_be_c60 = (atoms.size() == 60);
    bool all_carbons = true;
    
    for (const auto& atom : atoms) {
        if (atom.type.substr(0, 1) != "C") {
            all_carbons = false;
            break;
        }
    }
    
    bool is_c60 = might_be_c60 && all_carbons;
    
    // Determine typical bond lengths based on element types
    for (size_t i = 0; i < atoms.size(); i++) {
        for (size_t j = i + 1; j < atoms.size(); j++) {
            // Calculate distance between atoms
            Vec3 diff = atoms[i].position - atoms[j].position;
            double dist = diff.norm();
            
            // Get expected bond length between these elements
            double expected_bond_length = get_expected_bond_length(atoms[i].type, atoms[j].type);
            
            // Use a tolerance to determine if these atoms are bonded
            // For C60, use tighter tolerances to correctly identify the bonding pattern
            double tolerance = is_c60 ? 1.2 : 1.3;
            
            if (dist < tolerance * expected_bond_length) {
                connectivity[i].push_back(j);
                connectivity[j].push_back(i);
                bond_distances.push_back(dist);
            }
        }
    }
    
    // Special case for C60 fullerene
    if (is_c60) {
        // In C60, each carbon should have exactly 3 bonds
        bool valid_c60_connectivity = true;
        for (const auto& neighbors : connectivity) {
            if (neighbors.size() != 3) {
                valid_c60_connectivity = false;
                break;
            }
        }
        
        if (valid_c60_connectivity) {
            // Apply C60-specific refinements
            for (auto& atom : atoms) {
                if (atom.type.substr(0, 1) == "C") {
                    atom.type = "C_R";  // Set all carbons to aromatic
                }
            }
            return;  // Skip further refinement
        }
    }
    
    // For other molecules, refine types based on connectivity
    for (size_t i = 0; i < atoms.size(); i++) {
        std::string element = atoms[i].type.substr(0, 1);
        
        // Only update carbon and nitrogen types based on connectivity
        if (element == "C" || element == "N") {
            int conn = connectivity[i].size();
            
            if (element == "C") {
                if (conn <= 1) {
                    atoms[i].type = "C_1";  // Linear carbon
                } else if (conn == 2) {
                    atoms[i].type = "C_2";  // sp2 carbon (unless triple bond)
                } else if (conn == 3) {
                    atoms[i].type = "C_2";  // Likely sp2
                } else {
                    atoms[i].type = "C_3";  // sp3 (tetrahedral)
                }
            } else if (element == "N") {
                if (conn <= 1) {
                    atoms[i].type = "N_1";  // Linear nitrogen
                } else if (conn == 2) {
                    atoms[i].type = "N_2";  // sp2 nitrogen
                } else {
                    atoms[i].type = "N_3";  // sp3 nitrogen
                }
            }
        }
        
        // Special case for oxygen
        if (element == "O") {
            int conn = connectivity[i].size();
            if (conn == 1) {
                atoms[i].type = "O_2";  // Most likely sp2 (carbonyl)
            } else {
                atoms[i].type = "O_3";  // sp3 oxygen
            }
        }
    }
    
    // Additional refinement could be performed based on bond angles, etc.
}

// Get expected bond length between two atoms
double XYZParser::get_expected_bond_length(const std::string& type1, const std::string& type2) {
    // Get element symbols
    std::string elem1 = type1.substr(0, type1.find('_'));
    if (elem1.empty()) elem1 = type1;
    
    std::string elem2 = type2.substr(0, type2.find('_'));
    if (elem2.empty()) elem2 = type2;
    
    // Common covalent distances (Å)
    static const std::unordered_map<std::string, double> covalent_radii = {
        {"H", 0.31}, {"He", 0.28}, {"Li", 1.28}, {"Be", 0.96}, {"B", 0.84},
        {"C", 0.76}, {"N", 0.71}, {"O", 0.66}, {"F", 0.57}, {"Ne", 0.58},
        {"Na", 1.66}, {"Mg", 1.41}, {"Al", 1.21}, {"Si", 1.11}, {"P", 1.07},
        {"S", 1.05}, {"Cl", 1.02}, {"Ar", 1.06}, {"K", 2.03}, {"Ca", 1.76},
        {"Sc", 1.70}, {"Ti", 1.60}, {"V", 1.53}, {"Cr", 1.39}, {"Mn", 1.39},
        {"Fe", 1.32}, {"Co", 1.26}, {"Ni", 1.24}, {"Cu", 1.32}, {"Zn", 1.22},
        {"Ga", 1.22}, {"Ge", 1.20}, {"As", 1.19}, {"Se", 1.20}, {"Br", 1.20},
        {"Kr", 1.16}, {"Rb", 2.20}, {"Sr", 1.95}, {"Y", 1.90}, {"Zr", 1.75},
        {"Nb", 1.64}, {"Mo", 1.54}, {"Tc", 1.47}, {"Ru", 1.46}, {"Rh", 1.42},
        {"Pd", 1.39}, {"Ag", 1.45}, {"Cd", 1.44}, {"In", 1.42}, {"Sn", 1.39},
        {"Sb", 1.39}, {"Te", 1.38}, {"I", 1.39}, {"Xe", 1.40}, {"Cs", 2.44},
        {"Ba", 2.15}, {"La", 2.07}, {"Ce", 2.04}, {"Pr", 2.03}, {"Nd", 2.01},
        {"Pm", 1.99}, {"Sm", 1.98}, {"Eu", 1.98}, {"Gd", 1.96}, {"Tb", 1.94},
        {"Dy", 1.92}, {"Ho", 1.92}, {"Er", 1.89}, {"Tm", 1.90}, {"Yb", 1.87},
        {"Lu", 1.87}, {"Hf", 1.75}, {"Ta", 1.70}, {"W", 1.62}, {"Re", 1.51},
        {"Os", 1.44}, {"Ir", 1.41}, {"Pt", 1.36}, {"Au", 1.36}, {"Hg", 1.32},
        {"Tl", 1.45}, {"Pb", 1.46}, {"Bi", 1.48}, {"Po", 1.40}, {"At", 1.50},
        {"Rn", 1.50}, {"Fr", 2.60}, {"Ra", 2.21}, {"Ac", 2.15}, {"Th", 2.06},
        {"Pa", 2.00}, {"U", 1.96}, {"Np", 1.90}, {"Pu", 1.87}, {"Am", 1.80},
        {"Cm", 1.69}
    };
    
    // Apply hybridization correction factors
    double hybridization_factor1 = 1.0;
    double hybridization_factor2 = 1.0;
    
    // Adjust based on hybridization
    if (type1.find("_1") != std::string::npos) hybridization_factor1 = 0.95;  // sp
    if (type1.find("_2") != std::string::npos || type1.find("_R") != std::string::npos) hybridization_factor1 = 0.97;  // sp2
    if (type2.find("_1") != std::string::npos) hybridization_factor2 = 0.95;  // sp
    if (type2.find("_2") != std::string::npos || type2.find("_R") != std::string::npos) hybridization_factor2 = 0.97;  // sp2
    
    // Calculate expected bond length from covalent radii
    double radius1 = covalent_radii.count(elem1) ? covalent_radii.at(elem1) : 1.5;  // Default if unknown
    double radius2 = covalent_radii.count(elem2) ? covalent_radii.at(elem2) : 1.5;  // Default if unknown
    
    // Apply hybridization factors and return sum of adjusted radii
    return (radius1 * hybridization_factor1) + (radius2 * hybridization_factor2);
}

// Calculate distance between two atoms, handling periodic boundary conditions if applicable
double XYZParser::calculate_distance(const Vec3& pos1, const Vec3& pos2, const Vec3* cell_dimensions) {
    Vec3 diff = pos1 - pos2;
    
    // Apply periodic boundary conditions if cell dimensions are provided
    if (cell_dimensions) {
        for (int i = 0; i < 3; i++) {
            // Minimum image convention
            if ((*cell_dimensions)[i] > 0) {  // Only apply PBC along dimensions with positive length
                diff[i] -= (*cell_dimensions)[i] * std::round(diff[i] / (*cell_dimensions)[i]);
            }
        }
    }
    
    return diff.norm();
}

// Calculate bond angle between three atoms, handling periodic boundary conditions
double XYZParser::calculate_angle(const Vec3& pos1, const Vec3& pos2, const Vec3& pos3, 
                                 const Vec3* cell_dimensions) {
    Vec3 v1 = pos1 - pos2;
    Vec3 v2 = pos3 - pos2;
    
    // Apply periodic boundary conditions if cell dimensions are provided
    if (cell_dimensions) {
        for (int i = 0; i < 3; i++) {
            if ((*cell_dimensions)[i] > 0) {
                v1[i] -= (*cell_dimensions)[i] * std::round(v1[i] / (*cell_dimensions)[i]);
                v2[i] -= (*cell_dimensions)[i] * std::round(v2[i] / (*cell_dimensions)[i]);
            }
        }
    }
    
    // Calculate angle in radians
    double v1_norm = v1.norm();
    double v2_norm = v2.norm();
    
    if (v1_norm < 1e-10 || v2_norm < 1e-10) {
        return 0.0;  // Avoid division by zero
    }
    
    double cos_angle = v1.dot(v2) / (v1_norm * v2_norm);
    
    // Ensure cos_angle is in the valid range [-1, 1]
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
    
    return std::acos(cos_angle);
}

} // namespace UFF
