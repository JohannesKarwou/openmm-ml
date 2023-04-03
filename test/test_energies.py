from openmm.app import (
    StateDataReporter,
    DCDReporter,
    CharmmPsfFile,
    CharmmCrdFile,
    CharmmParameterSet,
    PME,
    CutoffPeriodic,
    NoCutoff,
    Simulation,
    HBonds,
)
from openmm import (
    unit,
    MonteCarloBarostat,
    LangevinIntegrator,
    Platform,
    NonbondedForce,
)
import numpy as np
import openmm

import openmm.unit as unit
from openmmml.mlpotential import MLPotential


import logging

logger = logging.getLogger(__name__)


def charmm_waterbox():

    psf = CharmmPsfFile(f"test/charmm_waterbox.psf")
    crd = CharmmCrdFile(f"test/charmm_waterbox.crd")
    params = CharmmParameterSet("test/toppar_water_ions.str")

    psf.setBox(
        15.5516 * unit.angstrom, 15.5516 * unit.angstrom, 15.5516 * unit.angstrom
    )

    return psf, crd, params


def charmm_params():
    temp = 303.15
    dt = 0.001  # 1 fs
    r_off = 0.65
    step = 1000000

    integrator = LangevinIntegrator(
        temp * unit.kelvin, 1 / unit.picosecond, dt * unit.picoseconds
    )
    platform = Platform.getPlatformByName("CUDA")
    prop = dict(CudaPrecision="mixed")

    return temp, dt, r_off, step, integrator, platform, prop


def mm_system_charmm():
    psf, crd, params = charmm_waterbox()
    temp, dt, r_off, step, integrator, platform, prop = charmm_params()

    system = psf.createSystem(
        params,
        nonbondedMethod=CutoffPeriodic,
        nonbondedCutoff=r_off * unit.nanometers,
        temperature=temp,
    )

    return system


def test_creating_mm_system_charmm():

    psf, crd, params = charmm_waterbox()
    temp, dt, r_off, step, integrator, platform, prop = charmm_params()

    system = mm_system_charmm()

    simulation = Simulation(psf.topology, system, integrator, platform, prop)

    simulation.context.setPositions(crd.positions)
    simulation.context.setVelocitiesToTemperature(303.15 * unit.kelvin)

    i = 0
    for force in system.getForces():
        force.setForceGroup(i)
        i += 1

    names = []
    values = []
    for f in system.getForces():
        group = f.getForceGroup()
        state = simulation.context.getState(getEnergy=True, groups={group})
        names.append(f"{f.getName()}")
        energy = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        values.append(energy)
        if isinstance(f, NonbondedForce):
            mm_nonbonded_energy = energy
            assert np.isclose(mm_nonbonded_energy, -1142.246)

    return mm_nonbonded_energy


def test_created_mixed_charmm_system():

    lamb = 1
    psf, crd, params = charmm_waterbox()
    temp, dt, r_off, step, integrator, platform, prop = charmm_params()
    ml_atoms = [atom.index for atom in psf.topology.atoms()]
    potential = MLPotential("ani2x")

    system = potential.createMixedSystem(
        psf.topology,
        mm_system_charmm(),
        ml_atoms,
        interpolate=True,
        implementation="torchani",
        removeConstraints=False,
    )

    #####  with torchani this is only working in the openmmml-test environment ###
    # barostat = MonteCarloBarostat(1.0 * unit.bar, temp * unit.kelvin)
    # system.addForce(barostat)

    simulation = Simulation(psf.topology, system, integrator, platform, prop)

    simulation.context.setPositions(crd.positions)
    simulation.context.setVelocitiesToTemperature(303.15 * unit.kelvin)
    simulation.context.setParameter("lambda_interpolate", lamb)

    i = 0
    for force in system.getForces():
        force.setForceGroup(i)
        i += 1

    for f in system.getForces():
        group = f.getForceGroup()
        state = simulation.context.getState(getEnergy=True, groups={group})
        if isinstance(f, openmm.CustomCVForce):
            for i in range(f.getNumCollectiveVariables()):
                cv = f.getCollectiveVariable(i)
                state = f.getInnerContext(simulation.context).getState(
                    getEnergy=True, groups={i}
                )
                energy = state.getPotentialEnergy().value_in_unit(
                    unit.kilocalorie_per_mole
                )
                logger.info(
                    f"CV name: {f.getCollectiveVariableName(i)}, force name {cv.getName()},{f.getForceGroup()}, {energy}"
                )

    if step > 0:
        print("\nMD run: %s steps" % step)
        simulation.reporters.append(
            DCDReporter(f"equi_{lamb}.dcd", 100, enforcePeriodicBox=True)
        )
        simulation.reporters.append(
            StateDataReporter(
                f"properties_{lamb}.csv",
                reportInterval=1000,
                step=True,
                potentialEnergy=True,
                density=True,
                speed=True,
                separator="\t",
            )
        )
        simulation.step(step)
