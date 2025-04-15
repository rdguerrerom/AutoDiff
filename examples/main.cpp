#include "uff.hpp"
#include <iostream>

#include "uff.hpp"
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <xyz_file>" << std::endl;
            return 1;
        }
        
        // Parse XYZ file to get atoms
        auto atoms = UFF::XYZParser::parse(argv[1]);
        std::cout << "Loaded " << atoms.size() << " atoms from " << argv[1] << std::endl;
        
        // Detect bonds, angles, torsions, and inversions with proper parameters for fullerenes
        double bond_tolerance = 1.1; // Optimized for fullerene detection
        auto bonds = UFF::TopologyBuilder::detect_bonds(atoms, bond_tolerance);
        auto angles = UFF::TopologyBuilder::detect_angles(bonds);
        auto torsions = UFF::TopologyBuilder::detect_torsions(bonds);
        auto inversions = UFF::TopologyBuilder::detect_inversions(atoms, bonds);
        
        // Output topology statistics for debugging
        std::cout << "Detected " << bonds.size() << " bonds" << std::endl;
        std::cout << "Detected " << angles.size() << " angles" << std::endl;
        std::cout << "Detected " << torsions.size() << " torsions" << std::endl;
        std::cout << "Detected " << inversions.size() << " inversions" << std::endl;
        
        // Create energy calculator and compute energy
        UFF::EnergyCalculator calculator(std::move(atoms), bonds, angles, torsions, inversions);
        const double energy = calculator.calculate();
        
        std::cout << "UFF Energy: " << energy << " kcal/mol" << std::endl;
        std::cout << "           " << energy * UFF::Constants::kcal_to_eV << " eV" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
