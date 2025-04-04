#include "uff_simulator.h"
#include "xyz_parser.h"
#include <iostream>

int main() {
    // Read C₆₀ structure from XYZ file
    std::vector<Atom> system = XYZParser::parse("c60.xyz");
    
    // Add noble gas atom (e.g., Ar)
    system.emplace_back("Ar", Vec3(0, 0, 0));

    // Initialize simulator
    UFFSimulator simulator;
    
    // Generate bonds and angles automatically
    std::vector<UFFBond> bonds = XYZParser::generateBonds(system);
    for (const auto& bond : bonds) simulator.addBond(bond);

    // Compute energy
    double energy = simulator.computeEnergy(system);
    std::cout << "Potential energy of X@C₆₀: " << energy << " kcal/mol" << std::endl;

    return 0;
}
