//
// Author:  Rafael Rangel, rrangel@cimne.upc.edu
// Date:    November 2021
//

// System includes

// Project includes
#include "graph_utilities.h"

// External includes

namespace Kratos {
  //-----------------------------------------------------------------------------------------------------------------------
  GraphUtilities::GraphUtilities()
  {
    mGraph_ParticleTempMin               = false;
    mGraph_ParticleTempMax               = false;
    mGraph_ParticleTempAvg               = false;
    mGraph_ParticleTempDev               = false;
    mGraph_ModelTempAvg                  = false;
    mGraph_ParticleHeatFluxContributions = false;
  }

  GraphUtilities::~GraphUtilities() {}

  //-----------------------------------------------------------------------------------------------------------------------
  void GraphUtilities::ExecuteInitialize(bool ParticleTempMin,
                                         bool ParticleTempMax,
                                         bool ParticleTempAvg,
                                         bool ParticleTempDev,
                                         bool ModelTempAvg,
                                         bool ParticleHeatFluxContributions)
  {
    KRATOS_TRY

    // Set member flags
    mGraph_ParticleTempMin               = ParticleTempMin;
    mGraph_ParticleTempMax               = ParticleTempMax;
    mGraph_ParticleTempAvg               = ParticleTempAvg;
    mGraph_ParticleTempDev               = ParticleTempDev;
    mGraph_ModelTempAvg                  = ModelTempAvg;
    mGraph_ParticleHeatFluxContributions = ParticleHeatFluxContributions;

    // Open files
    if (mGraph_ParticleTempMin) {
      mFile_ParticleTempMin.open("graph_particle_temp_min.txt", std::ios::out);
      KRATOS_ERROR_IF_NOT(mFile_ParticleTempMin) << "Could not open graph file for minimum particle temperature!" << std::endl;
      mFile_ParticleTempMin << "TIME STEP | TIME | MIN PARTICLE TEMPERATURE" << std::endl;
    }
    if (mGraph_ParticleTempMax) {
      mFile_ParticleTempMax.open("graph_particle_temp_max.txt", std::ios::out);
      KRATOS_ERROR_IF_NOT(mFile_ParticleTempMax) << "Could not open graph file for maximum particle temperature!" << std::endl;
      mFile_ParticleTempMax << "TIME STEP | TIME | MAX PARTICLE TEMPERATURE" << std::endl;
    }
    if (mGraph_ParticleTempAvg) {
      mFile_ParticleTempAvg.open("graph_particle_temp_avg.txt", std::ios::out);
      KRATOS_ERROR_IF_NOT(mFile_ParticleTempAvg) << "Could not open graph file for average particle temperature!" << std::endl;
      mFile_ParticleTempAvg << "TIME STEP | TIME | AVERAGE PARTICLE TEMPERATURE" << std::endl;
    }
    if (mGraph_ParticleTempDev) {
      mFile_ParticleTempDev.open("graph_particle_temp_dev.txt", std::ios::out);
      KRATOS_ERROR_IF_NOT(mFile_ParticleTempDev) << "Could not open graph file for deviation of particle temperature!" << std::endl;
      mFile_ParticleTempDev << "TIME STEP | TIME | PARTICLE TEMPERATURE STANDARD DEVIATION" << std::endl;
    }
    if (mGraph_ModelTempAvg) {
      mFile_ModelTempAvg.open("graph_model_temp_avg.txt", std::ios::out);
      KRATOS_ERROR_IF_NOT(mFile_ModelTempAvg) << "Could not open graph file for average model temperature!" << std::endl;
      mFile_ModelTempAvg << "TIME STEP | TIME | AVERAGE MODEL TEMPERATURE" << std::endl;
    }
    if (mGraph_ParticleHeatFluxContributions) {
      mFile_ParticleHeatFluxContributions.open("graph_flux_contributions.txt", std::ios::out);
      KRATOS_ERROR_IF_NOT(mFile_ParticleHeatFluxContributions) << "Could not open graph file for heat flux contributions!" << std::endl;
      mFile_ParticleHeatFluxContributions << "TIME STEP | TIME | DIRECT CONDUCTION | INDIRECT CONDUCTION | RADIATION | FRICTION | CONVECTION | SURFACE PRESCRIBED | VOLUME PRESCRIBED" << std::endl;
    }

    KRATOS_CATCH("")
  }

  //-----------------------------------------------------------------------------------------------------------------------
  void GraphUtilities::ExecuteFinalizeSolutionStep(ModelPart& r_modelpart)
  {
    KRATOS_TRY

    const ProcessInfo& r_process_info = r_modelpart.GetProcessInfo();
    if (!r_process_info[IS_TIME_TO_PRINT])
      return;

    // Initialize results
    int    num_of_particles                    = r_modelpart.NumberOfElements();
    int    time_step                           = r_process_info[TIME_STEPS];
    double time                                = r_process_info[TIME];
    double total_vol                           = 0.0;
    double particle_temp_min                   = DBL_MAX;
    double particle_temp_max                   = DBL_MIN;
    double particle_temp_avg                   = 0.0;
    double particle_temp_dev                   = 0.0;
    double model_temp_avg                      = 0.0;
    double particle_flux_conducdir_ratio_avg   = 0.0;
    double particle_flux_conducindir_ratio_avg = 0.0;
    double particle_flux_rad_ratio_avg         = 0.0;
    double particle_flux_fric_ratio_avg        = 0.0;
    double particle_flux_conv_ratio_avg        = 0.0;
    double particle_flux_prescsurf_ratio_avg   = 0.0;
    double particle_flux_prescvol_ratio_avg    = 0.0;

    #pragma omp parallel for schedule(dynamic, 100)
    for (int i = 0; i < num_of_particles; i++) {
      ModelPart::ElementsContainerType::iterator it = r_modelpart.GetCommunicator().LocalMesh().Elements().ptr_begin() + i;
      ThermalSphericParticle<SphericParticle>& particle = dynamic_cast<ThermalSphericParticle<SphericParticle>&> (*it);

      // Accumulate total volume
      double vol = particle.CalculateVolume();
      total_vol += vol;

      // Accumulate temperature results (min, max, avg, dev)
      double temp = particle.GetGeometry()[0].FastGetSolutionStepValue(TEMPERATURE);
      if (temp < particle_temp_min) particle_temp_min = temp;
      if (temp > particle_temp_max) particle_temp_max = temp;
      particle_temp_avg += temp;
      particle_temp_dev += temp * temp;
      model_temp_avg    += temp * vol;

      if (mGraph_ParticleHeatFluxContributions) {
        // Get absolute value of particle heat transfer mechanisms
        double flux_conducdir   = fabs(particle.mConductionDirectHeatFlux);
        double flux_conducindir = fabs(particle.mConductionIndirectHeatFlux);
        double flux_rad         = fabs(particle.mRadiationHeatFlux);
        double flux_fric        = fabs(particle.mFrictionHeatFlux);
        double flux_conv        = fabs(particle.mConvectionHeatFlux);
        double flux_prescsurf   = fabs(particle.mPrescribedHeatFluxSurface);
        double flux_prescvol    = fabs(particle.mPrescribedHeatFluxVolume);
        double flux_total       = flux_conducdir + flux_conducindir + flux_rad + flux_fric + flux_conv + flux_prescsurf + flux_prescvol;

        // Compute relative contribution of each heat transfer mechanism for current particle
        if (flux_total != 0.0) {
          particle_flux_conducdir_ratio_avg   += flux_conducdir   / flux_total;
          particle_flux_conducindir_ratio_avg += flux_conducindir / flux_total;
          particle_flux_rad_ratio_avg         += flux_rad         / flux_total;
          particle_flux_fric_ratio_avg        += flux_fric        / flux_total;
          particle_flux_conv_ratio_avg        += flux_conv        / flux_total;
          particle_flux_prescsurf_ratio_avg   += flux_prescsurf   / flux_total;
          particle_flux_prescvol_ratio_avg    += flux_prescvol    / flux_total;
        }
      }
    }

    // Compute temperature results (avg, dev)
    particle_temp_avg /= num_of_particles;
    model_temp_avg    /= total_vol;
    particle_temp_dev  = std::max(0.0, particle_temp_dev / num_of_particles - particle_temp_avg * particle_temp_avg);

    // Compute average of relative contribution of each heat transfer mechanism
    if (mGraph_ParticleHeatFluxContributions) {
      particle_flux_conducdir_ratio_avg   /= num_of_particles;
      particle_flux_conducindir_ratio_avg /= num_of_particles;
      particle_flux_rad_ratio_avg         /= num_of_particles;
      particle_flux_fric_ratio_avg        /= num_of_particles;
      particle_flux_conv_ratio_avg        /= num_of_particles;
      particle_flux_prescsurf_ratio_avg   /= num_of_particles;
      particle_flux_prescvol_ratio_avg    /= num_of_particles;
    }

    // Write results to files
    if (mFile_ParticleTempMin.is_open())
      mFile_ParticleTempMin << time_step << " " << time << " " << particle_temp_min << std::endl;
    if (mFile_ParticleTempMax.is_open())
      mFile_ParticleTempMax << time_step << " " << time << " " << particle_temp_max << std::endl;
    if (mFile_ParticleTempAvg.is_open())
      mFile_ParticleTempAvg << time_step << " " << time << " " << particle_temp_avg << std::endl;
    if (mFile_ParticleTempDev.is_open())
      mFile_ParticleTempDev << time_step << " " << time << " " << particle_temp_dev << std::endl;
    if (mFile_ModelTempAvg.is_open())
      mFile_ModelTempAvg << time_step << " " << time << " " << model_temp_avg << std::endl;
    if (mFile_ParticleHeatFluxContributions.is_open())
      mFile_ParticleHeatFluxContributions << time_step << " " << time << " " << particle_flux_conducdir_ratio_avg << " " << particle_flux_conducindir_ratio_avg << " " << particle_flux_rad_ratio_avg << " " << particle_flux_fric_ratio_avg << " " << particle_flux_conv_ratio_avg << " " << particle_flux_prescsurf_ratio_avg << " " << particle_flux_prescvol_ratio_avg << std::endl;

    KRATOS_CATCH("")
  }

  //-----------------------------------------------------------------------------------------------------------------------
  void GraphUtilities::ExecuteFinalize(void)
  {
    KRATOS_TRY

    // Close files
    if (mFile_ParticleTempMin.is_open())               mFile_ParticleTempMin.close();
    if (mFile_ParticleTempMax.is_open())               mFile_ParticleTempMax.close();
    if (mFile_ParticleTempAvg.is_open())               mFile_ParticleTempAvg.close();
    if (mFile_ParticleTempDev.is_open())               mFile_ParticleTempDev.close();
    if (mFile_ModelTempAvg.is_open())                  mFile_ModelTempAvg.close();
    if (mFile_ParticleHeatFluxContributions.is_open()) mFile_ParticleHeatFluxContributions.close();

    KRATOS_CATCH("")
  }

} // namespace Kratos