#include "uff.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <xyz file>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    
    try {
        // Parse the XYZ file
        auto atoms = UFF::XYZParser::parse(filename);
        
        // Automatically detect topology
        auto bonds = UFF::TopologyBuilder::detect_bonds(atoms);
        auto angles = UFF::TopologyBuilder::detect_angles(bonds);
        auto torsions = UFF::TopologyBuilder::detect_torsions(bonds);
        auto inversions = UFF::TopologyBuilder::detect_inversions(atoms, bonds);
        
        // Create energy calculator
        UFF::EnergyCalculator calculator(atoms, bonds, angles, torsions, inversions);
        
        // Calculate energy
        double energy = calculator.calculate();
        
        std::cout << "Total energy: " << energy << " kcal/mol" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
