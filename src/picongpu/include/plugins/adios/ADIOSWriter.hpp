/**
 * Copyright 2014-2015 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pthread.h>
#include <cassert>
#include <sstream>
#include <list>
#include <vector>

#include "types.h"
#include "simulation_types.hpp"
#include "plugins/adios/ADIOSWriter.def"

#include "particles/frame_types.hpp"

#include <adios.h>

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldJ.hpp"
#include "fields/FieldTmp.hpp"
#include "particles/particleFilter/FilterFactory.hpp"
#include "particles/particleFilter/PositionFilter.hpp"
#include "particles/operations/CountParticles.hpp"

#include "dataManagement/DataConnector.hpp"
#include "mappings/simulation/GridController.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "dimensions/GridLayout.hpp"
#include "pluginSystem/PluginConnector.hpp"
#include "simulationControl/MovingWindow.hpp"
#include "math/Vector.hpp"

#include "plugins/ILightweightPlugin.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>

#include <boost/type_traits.hpp>

#include "plugins/adios/WriteSpecies.hpp"
#include "plugins/adios/ADIOSCountParticles.hpp"


namespace picongpu
{

namespace adios
{

using namespace PMacc;



namespace po = boost::program_options;

template <unsigned DIM>
int64_t defineAdiosVar(int64_t group_id,
                       const char * name,
                       const char * path,
                       enum ADIOS_DATATYPES type,
                       PMacc::math::UInt64<DIM> dimensions,
                       PMacc::math::UInt64<DIM> globalDimensions,
                       PMacc::math::UInt64<DIM> offset,
                       bool compression,
                       std::string compressionMethod)
{
    /* disable compression if this rank writes no data */
    bool canCompress = true;
    for (size_t i = 0; i < DIM; ++i)
    {
        if (dimensions[i] == 0 || globalDimensions[i] == 0)
        {
            canCompress = false;
        }
    }

    int64_t var_id = 0;
    if ((DIM == 1) && (globalDimensions.productOfComponents() == 1)) {
        /* scalars need empty size strings */
        var_id = adios_define_var(
            group_id, name, path, type, 0, 0, 0);
    } else {
        var_id = adios_define_var(
            group_id, name, path, type,
            dimensions.toString(",", "").c_str(),
            globalDimensions.toString(",", "").c_str(),
            offset.toString(",", "").c_str());
    }

    if (compression && canCompress)
    {
        /* enable zlib compression for variable, default compression level */
#if(ADIOS_TRANSFORMS==1)
        adios_set_transform(var_id, compressionMethod.c_str());
#endif
    }

    log<picLog::INPUT_OUTPUT > ("ADIOS: Defined varID=%1% for '%2%'") %
                var_id % std::string(name);
    return var_id;
}

/**
 * Writes simulation data to adios files.
 * Implements the ILightweightPlugin interface.
 */
class ADIOSWriter : public ILightweightPlugin
{
public:

    /* filter particles by global position*/
    typedef bmpl::vector< typename GetPositionFilter<simDim>::type > usedFilters;
    typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;

private:

    /* filter is a rule which describes which particles should be copied to host*/
    MyParticleFilter filter;

    template<typename UnitType>
    static std::vector<double> createUnit(UnitType unit, uint32_t numComponents)
    {
        std::vector<double> tmp(numComponents);
        for (uint i = 0; i < numComponents; ++i)
            tmp[i] = unit[i];
        return tmp;
    }

    /**
     * Write calculated fields to adios file.
     */
    template< typename T >
    struct GetFields
    {
    private:
        typedef typename T::ValueType ValueType;
        typedef typename GetComponentsType<ValueType>::type ComponentType;

        static std::vector<double> getUnit()
        {
            typedef typename T::UnitValueType UnitType;
            UnitType unit = T::getUnit();
            return createUnit(unit, T::numComponents);
        }

    public:

        HDINLINE void operator()(ThreadParams* params)
        {
#ifndef __CUDA_ARCH__
            DataConnector &dc = Environment<simDim>::get().DataConnector();

            T* field = &(dc.getData<T > (T::getName()));
            params->gridLayout = field->getGridLayout();

            PICToAdios<ComponentType> adiosType;
            writeField(params,
                       sizeof(ComponentType),
                       adiosType.type,
                       GetNComponents<ValueType>::value,
                       T::getName(),
                       getUnit(),
                       field->getHostDataBox().getPointer());

            dc.releaseData(T::getName());
#endif
        }

    };

    /** Calculate FieldTmp with given solver and particle species
     * and write them to adios.
     *
     * FieldTmp is calculated on device and than dumped to adios.
     */
    template< typename Solver, typename Species >
    struct GetFields<FieldTmpOperation<Solver, Species> >
    {

        /*
         * This is only a wrapper function to allow disable nvcc warnings.
         * Warning: calling a __host__ function from __host__ __device__
         * function.
         * Use of PMACC_NO_NVCC_HDWARNING is not possible if we call a virtual
         * method inside of the method were we disable the warnings.
         * Therefore we create this method and call a new method were we can
         * call virtual functions.
         */
        PMACC_NO_NVCC_HDWARNING
        HDINLINE void operator()(ThreadParams* tparam)
        {
            this->operator_impl(tparam);
        }
    private:
        typedef typename FieldTmp::ValueType ValueType;
        typedef typename GetComponentsType<ValueType>::type ComponentType;

        /** Create a name for the adios identifier.
         */
        static std::string getName()
        {
            std::stringstream str;
            str << Solver().getName();
            str << "_";
            str << Species::FrameType::getName();
            return str.str();
        }

        /** Get the unit for the result from the solver*/
        static std::vector<double> getUnit()
        {
            typedef typename FieldTmp::UnitValueType UnitType;
            UnitType unit = FieldTmp::getUnit<Solver>();
            const uint32_t components = GetNComponents<ValueType>::value;
            return createUnit(unit, components);
        }

        HINLINE void operator_impl(ThreadParams* params)
        {
            DataConnector &dc = Environment<>::get().DataConnector();

            /*## update field ##*/

            /*load FieldTmp without copy data to host*/
            FieldTmp* fieldTmp = &(dc.getData<FieldTmp > (FieldTmp::getName(), true));
            /*load particle without copy particle data to host*/
            Species* speciesTmp = &(dc.getData<Species >(Species::FrameType::getName(), true));

            fieldTmp->getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.0));
            /*run algorithm*/
            fieldTmp->computeValue < CORE + BORDER, Solver > (*speciesTmp, params->currentStep);

            EventTask fieldTmpEvent = fieldTmp->asyncCommunication(__getTransactionEvent());
            __setTransactionEvent(fieldTmpEvent);
            /* copy data to host that we can write same to disk*/
            fieldTmp->getGridBuffer().deviceToHost();
            dc.releaseData(Species::FrameType::getName());
            /*## finish update field ##*/

            const uint32_t components = GetNComponents<ValueType>::value;
            PICToAdios<ComponentType> adiosType;

            params->gridLayout = fieldTmp->getGridLayout();
            /*write data to ADIOS file*/
            writeField(params,
                       sizeof(ComponentType),
                       adiosType.type,
                       components,
                       getName(),
                       getUnit(),
                       fieldTmp->getHostDataBox().getPointer());

            dc.releaseData(FieldTmp::getName());

        }

    };

    static void defineFieldVar(ThreadParams* params,
        uint32_t nComponents, ADIOS_DATATYPES adiosType, const std::string name)
    {
        const std::string name_lookup_tpl[] = {"x", "y", "z", "w"};

        for (uint32_t c = 0; c < nComponents; c++)
        {
            std::stringstream datasetName;
            datasetName << params->adiosBasePath << ADIOS_PATH_FIELDS << name;
            if (nComponents > 1)
                datasetName << "/" << name_lookup_tpl[c];

            /* define adios var for field, e.g. field_FieldE_y */
            const char* path = NULL;
            int64_t adiosFieldVarId = defineAdiosVar<simDim>(
                    params->adiosGroupHandle,
                    datasetName.str().c_str(),
                    path,
                    adiosType,
                    params->fieldsSizeDims,
                    params->fieldsGlobalSizeDims,
                    params->fieldsOffsetDims,
                    true,
                    params->adiosCompression);

            params->adiosFieldVarIds.push_back(adiosFieldVarId);
        }
    }

    /**
     * Collect field sizes to set adios group size.
     */
    template< typename T >
    struct CollectFieldsSizes
    {
    public:
        typedef typename T::ValueType ValueType;
        typedef typename GetComponentsType<ValueType>::type ComponentType;

        HDINLINE void operator()(ThreadParams* params)
        {
#ifndef __CUDA_ARCH__
            const uint32_t components = T::numComponents;

            // adios buffer size for this dataset (all components)
            uint64_t localGroupSize =
                    params->window.localDimensions.size.productOfComponents() *
                    sizeof(ComponentType) *
                    components;

            params->adiosGroupSize += localGroupSize;

            PICToAdios<ComponentType> adiosType;
            defineFieldVar(params, components, adiosType.type, T::getName());
#endif
        }
    };

    /**
     * Collect field sizes to set adios group size.
     * Specialization.
     */
    template< typename Solver, typename Species >
    struct CollectFieldsSizes<FieldTmpOperation<Solver, Species> >
    {
    public:

        PMACC_NO_NVCC_HDWARNING
        HDINLINE void operator()(ThreadParams* tparam)
        {
            this->operator_impl(tparam);
        }

   private:
        typedef typename FieldTmp::ValueType ValueType;
        typedef typename GetComponentsType<ValueType>::type ComponentType;

        /** Create a name for the adios identifier.
         */
        static std::string getName()
        {
            std::stringstream str;
            str << Solver().getName();
            str << "_";
            str << Species::FrameType::getName();
            return str.str();
        }

        HINLINE void operator_impl(ThreadParams* params)
        {
            const uint32_t components = GetNComponents<ValueType>::value;

            // adios buffer size for this dataset (all components)
            uint64_t localGroupSize =
                    params->window.localDimensions.size.productOfComponents() *
                    sizeof(ComponentType) *
                    components;

            params->adiosGroupSize += localGroupSize;

            PICToAdios<ComponentType> adiosType;
            defineFieldVar(params, components, adiosType.type, getName());
        }

    };

public:

    ADIOSWriter() :
    filename("simDataAdios"),
    checkpointFilename("adios_checkpoint"),
    restartFilename(""), /* set to checkpointFilename by default */
    /* select MPI method, #OSTs and #aggregators */
    mpiTransportParams(""),
    notifyPeriod(0)
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    virtual ~ADIOSWriter()
    {

    }

    void pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ("adios.period", po::value<uint32_t > (&notifyPeriod)->default_value(0),
             "enable ADIOS IO [for each n-th step]")
            ("adios.aggregators", po::value<uint32_t >
             (&mThreadParams.adiosAggregators)->default_value(0), "Number of aggregators [0 == number of MPI processes]")
            ("adios.ost", po::value<uint32_t > (&mThreadParams.adiosOST)->default_value(1),
             "Number of OST")
#if(ADIOS_TRANSFORMS==1)
            ("adios.compression", po::value<std::string >
             (&mThreadParams.adiosCompression)->default_value("none"),
             "ADIOS compression method (see 'adios_config -m' for help)")
#endif
            ("adios.file", po::value<std::string > (&filename)->default_value(filename),
             "ADIOS output file")
            ("adios.checkpoint-file", po::value<std::string > (&checkpointFilename),
             "Optional ADIOS checkpoint filename (prefix)")
            ("adios.restart-file", po::value<std::string > (&restartFilename),
             "adios restart filename (prefix)")
            /* 1,000,000 particles are around 3900 frames at 256 particles per frame
             * and match ~30MiB with typical picongpu particles.
             * The only reason why we use 1M particles per chunk is that we can get a
             * frame overflow in our memory manager if we process all particles in one kernel.
             **/
            ("adios.restart-chunkSize", po::value<uint32_t > (&restartChunkSize)->default_value(1000000),
             "Number of particles processed in one kernel call during restart to prevent frame count blowup");
    }

    std::string pluginGetName() const
    {
        return "ADIOSWriter";
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        this->cellDescription = cellDescription;
    }

    __host__ void notify(uint32_t currentStep)
    {
        notificationReceived(currentStep, false);
    }

    void checkpoint(uint32_t currentStep, const std::string checkpointDirectory)
    {
        this->checkpointDirectory = checkpointDirectory;

        notificationReceived(currentStep, true);
    }

    void restart(uint32_t restartStep, const std::string restartDirectory)
    {
        assert(1==2); // not yet implemented
    }

private:

    void endAdios()
    {
        /* Finalize adios library */
        ADIOS_CMD(adios_finalize(Environment<simDim>::get().GridController()
                .getCommunicator().getRank()));

        __deleteArray(mThreadParams.fieldBfr);
    }

    void beginAdios(const std::string adiosFilename)
    {
        std::stringstream full_filename;
        full_filename << adiosFilename << "_" << mThreadParams.currentStep << ".bp";

        mThreadParams.fullFilename = full_filename.str();
        mThreadParams.adiosFileHandle = ADIOS_INVALID_HANDLE;

        mThreadParams.fieldBfr = NULL;
        mThreadParams.fieldBfr = new float[mThreadParams.window.localDimensions.size.productOfComponents()];

        std::stringstream adiosPathBase;
        adiosPathBase << ADIOS_PATH_ROOT << mThreadParams.currentStep << "/";
        mThreadParams.adiosBasePath = adiosPathBase.str();

        ADIOS_CMD(adios_init_noxml(mThreadParams.adiosComm));
    }

    /**
     * Notification for dump or checkpoint received
     *
     * @param currentStep current simulation step
     * @param isCheckpoint checkpoint notification
     */
    void notificationReceived(uint32_t currentStep, bool isCheckpoint)
    {
        const PMacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
        mThreadParams.isCheckpoint = isCheckpoint;
        mThreadParams.currentStep = currentStep;
        mThreadParams.cellDescription = this->cellDescription;
        this->filter.setStatus(false);

        __getTransactionEvent().waitForFinished();

        std::string fname = filename;
        if (isCheckpoint)
        {
            /* if checkpointFilename is relative, prepend with checkpointDirectory */
            if (!boost::filesystem::path(checkpointFilename).has_root_path())
                fname = checkpointDirectory + std::string("/") + checkpointFilename;
            else
                fname = checkpointFilename;

            mThreadParams.window = MovingWindow::getInstance().getDomainAsWindow(currentStep);
        }
        else
        {
            mThreadParams.window = MovingWindow::getInstance().getWindow(currentStep);
        }

        for (uint32_t i = 0; i < simDim; ++i)
        {
            mThreadParams.localWindowToDomainOffset[i] = 0;
            if (mThreadParams.window.globalDimensions.offset[i] > localDomain.offset[i])
            {
                mThreadParams.localWindowToDomainOffset[i] =
                    mThreadParams.window.globalDimensions.offset[i] -
                    localDomain.offset[i];
            }
        }

        if (MovingWindow::getInstance().isSlidingWindowActive())
        {
            //enable filters for sliding window and configurate position filter
            this->filter.setStatus(true);

            this->filter.setWindowPosition(
                    mThreadParams.window.localDimensions.offset,
                    mThreadParams.window.localDimensions.size);
        }

        beginAdios(fname);

        writeAdios((void*) &mThreadParams, mpiTransportParams);

        endAdios();
    }

    void pluginLoad()
    {
        GridController<simDim> &gc = Environment<simDim>::get().GridController();
        /* It is important that we never change the mpi_pos after this point
         * because we get problems with the restart.
         * Otherwise we do not know which gpu must load the ghost parts around
         * the sliding window.
         */
        mpi_pos = gc.getPosition();
        mpi_size = gc.getGpuNodes();

        /* if number of aggregators is not set we use all mpi process as aggregator*/
        if( mThreadParams.adiosAggregators == 0 )
           mThreadParams.adiosAggregators=mpi_size.productOfComponents();

        if (notifyPeriod > 0)
        {
            Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
        }

        /* Initialize adios library */
        mThreadParams.adiosComm = MPI_COMM_NULL;
        MPI_CHECK(MPI_Comm_dup(gc.getCommunicator().getMPIComm(), &(mThreadParams.adiosComm)));
        mThreadParams.adiosBufferInitialized = false;

        /* select MPI method, #OSTs and #aggregators */
        std::stringstream strMPITransportParams;
        strMPITransportParams << "num_aggregators=" << mThreadParams.adiosAggregators
                              << ";num_ost=" << mThreadParams.adiosOST;
        mpiTransportParams = strMPITransportParams.str();

        loaded = true;
    }

    void pluginUnload()
    {
        if (notifyPeriod > 0)
        {
            if (mThreadParams.adiosComm != MPI_COMM_NULL)
            {
                MPI_CHECK(MPI_Comm_free(&(mThreadParams.adiosComm)));
            }
        }
    }

    static void writeField(ThreadParams *params, const uint32_t sizePtrType,
                           ADIOS_DATATYPES adiosType,
                           const uint32_t nComponents, const std::string name,
                           std::vector<double> unit, void *ptr)
    {
        log<picLog::INPUT_OUTPUT > ("ADIOS: write field: %1% %2% %3%") %
            name % nComponents % ptr;

        /* data to describe source buffer */
        GridLayout<simDim> field_layout = params->gridLayout;
        DataSpace<simDim> field_full = field_layout.getDataSpace();
        DataSpace<simDim> field_no_guard = params->window.localDimensions.size;
        DataSpace<simDim> field_guard = field_layout.getGuard() + params->localWindowToDomainOffset;

        /* write the actual field data */
        for (uint32_t d = 0; d < nComponents; d++)
        {
            const size_t plane_full_size = field_full[1] * field_full[0] * nComponents;
            const size_t plane_no_guard_size = field_no_guard[1] * field_no_guard[0];

            /* copy strided data from source to temporary buffer */
            for (int z = 0; z < field_no_guard[2]; ++z)
            {
                for (int y = 0; y < field_no_guard[1]; ++y)
                {
                    const size_t base_index_src =
                                (z + field_guard[2]) * plane_full_size +
                                (y + field_guard[1]) * field_full[0] * nComponents;

                    const size_t base_index_dst =
                                z * plane_no_guard_size +
                                y * field_no_guard[0];

                    for (int x = 0; x < field_no_guard[0]; ++x)
                    {
                        size_t index_src = base_index_src + (x + field_guard[0]) * nComponents + d;
                        size_t index_dst = base_index_dst + x;

                        params->fieldBfr[index_dst] = ((float*)ptr)[index_src];
                    }
                }
            }

            /* Write the actual field data. The id is on the front of the list. */
            if (params->adiosFieldVarIds.empty())
                throw std::runtime_error("Cannot write field (var id list is empty)");

            int64_t adiosFieldVarId = *(params->adiosFieldVarIds.begin());
            params->adiosFieldVarIds.pop_front();
            ADIOS_CMD(adios_write_byid(params->adiosFileHandle, adiosFieldVarId, params->fieldBfr));
        }
    }

    typedef PICToAdios<uint32_t> AdiosUInt32Type;
    typedef PICToAdios<float_X> AdiosFloatXType;
    typedef PICToAdios<double> AdiosDoubleType;

    /**
     * Define a single scalar meta attribute
     */
    static void defineMetaAttr(ThreadParams *threadParams, const char *name,
        enum ADIOS_DATATYPES adiosType)
    {
        threadParams->adiosMetaAttrVarIds.push_back(
            adios_define_var(threadParams->adiosGroupHandle,
                (threadParams->adiosBasePath + std::string(name)).c_str(), "",
                adiosType, 0, 0, 0));
    }

    /**
     * Define meta attributes
     */
    static void defineMetaAttributes(ThreadParams *threadParams)
    {
        AdiosUInt32Type adiosUInt32Type;
        AdiosFloatXType adiosFloatXType;
        AdiosDoubleType adiosDoubleType;

        /* iteration, sim_slides */
        defineMetaAttr(threadParams, "iteration", adiosUInt32Type.type);
        defineMetaAttr(threadParams, "sim_slides", adiosUInt32Type.type);
        threadParams->adiosGroupSize += 2 * sizeof(uint32_t);

        /* normed grid parameters */
        defineMetaAttr(threadParams, "delta_t", adiosFloatXType.type);
        defineMetaAttr(threadParams, "cell_width", adiosFloatXType.type);
        defineMetaAttr(threadParams, "cell_height", adiosFloatXType.type);
        if (simDim == DIM3)
            defineMetaAttr(threadParams, "cell_depth", adiosFloatXType.type);
        threadParams->adiosGroupSize += (1 + simDim) * sizeof(float_X);

        /* base units*/
        defineMetaAttr(threadParams, "unit_energy", adiosDoubleType.type);
        defineMetaAttr(threadParams, "unit_length", adiosDoubleType.type);
        defineMetaAttr(threadParams, "unit_speed", adiosDoubleType.type);
        defineMetaAttr(threadParams, "unit_time", adiosDoubleType.type);
        defineMetaAttr(threadParams, "unit_mass", adiosDoubleType.type);
        defineMetaAttr(threadParams, "unit_charge", adiosDoubleType.type);
        defineMetaAttr(threadParams, "unit_efield", adiosDoubleType.type);
        defineMetaAttr(threadParams, "unit_bfield", adiosDoubleType.type);
        threadParams->adiosGroupSize += 8 * sizeof(double);

        /* physical constants */
        defineMetaAttr(threadParams, "mue0", adiosFloatXType.type);
        defineMetaAttr(threadParams, "eps0", adiosFloatXType.type);
        threadParams->adiosGroupSize += 2 * sizeof(float_X);
    }

    /**
     * Write a single scalar meta attribute
     */
    static void writeMetaAttr(ThreadParams *threadParams, void *var)
    {
        int64_t var_id = threadParams->adiosMetaAttrVarIds.back();
        threadParams->adiosMetaAttrVarIds.pop_back();

        ADIOS_CMD(adios_write_byid(threadParams->adiosFileHandle, var_id, var));
    }

    /**
     * Write meta attributes
     * Attributes must be written in same order as defined using \see defineMetaAttributes
     *
     * @param threadParams parameters
     */
    static void writeMetaAttributes(ThreadParams *threadParams)
    {
        /* write number of slides to timestep in adios file */
        uint32_t slides = MovingWindow::getInstance().getSlideCounter(threadParams->currentStep);

        float_X varFloatX;
        double varDouble;

        /* write current iteration */
        writeMetaAttr(threadParams, &(threadParams->currentStep));

        /* write number of slides */
        writeMetaAttr(threadParams, &slides);

        /* write normed grid parameters */
        varFloatX = DELTA_T;
        writeMetaAttr(threadParams, &varFloatX);
        varFloatX = CELL_WIDTH;
        writeMetaAttr(threadParams, &varFloatX);
        varFloatX = CELL_HEIGHT;
        writeMetaAttr(threadParams, &varFloatX);
        if (simDim == DIM3)
        {
            varFloatX = CELL_DEPTH;
            writeMetaAttr(threadParams, &varFloatX);
        }

        /* write base units */
        varDouble = UNIT_ENERGY;
        writeMetaAttr(threadParams, &varDouble);
        varDouble = UNIT_LENGTH;
        writeMetaAttr(threadParams, &varDouble);
        varDouble = UNIT_SPEED;
        writeMetaAttr(threadParams, &varDouble);
        varDouble = UNIT_TIME;
        writeMetaAttr(threadParams, &varDouble);
        varDouble = UNIT_MASS;
        writeMetaAttr(threadParams, &varDouble);
        varDouble = UNIT_CHARGE;
        writeMetaAttr(threadParams, &varDouble);
        varDouble = UNIT_EFIELD;
        writeMetaAttr(threadParams, &varDouble);
        varDouble = UNIT_BFIELD;
        writeMetaAttr(threadParams, &varDouble);

        /* write physical constants */
        varFloatX = MUE0;
        writeMetaAttr(threadParams, &varFloatX);
        varFloatX = EPS0;
        writeMetaAttr(threadParams, &varFloatX);
    }

    static void *writeAdios(void *p_args, std::string mpiTransportParams)
    {

        // synchronize, because following operations will be blocking anyway
        ThreadParams *threadParams = (ThreadParams*) (p_args);
        threadParams->adiosGroupSize = 0;

        /* y direction can be negative for first gpu */
        DataSpace<simDim> particleOffset(Environment<simDim>::get().SubGrid().getLocalDomain().offset);
        particleOffset.y() -= threadParams->window.globalDimensions.offset.y();

        /* create adios group for fields without statistics */
        ADIOS_CMD(adios_declare_group(&(threadParams->adiosGroupHandle),
                ADIOS_GROUP_NAME,
                (threadParams->adiosBasePath + std::string("iteration")).c_str(),
                adios_flag_no));

        /* select MPI method, #OSTs and #aggregators */
        ADIOS_CMD(adios_select_method(threadParams->adiosGroupHandle,
                  "MPI_AGGREGATE", mpiTransportParams.c_str(), ""));

        /* write created variable values */
        for (uint32_t d = 0; d < simDim; ++d)
        {
            uint64_t offset = threadParams->window.localDimensions.offset[d];

            /* dimension 1 is y and is the direction of the moving window (if any) */
            if (1 == d)
                offset = std::max(0, threadParams->window.localDimensions.offset[1] -
                                     threadParams->window.globalDimensions.offset[1]);

            threadParams->fieldsSizeDims[d] = threadParams->window.localDimensions.size[d];
            threadParams->fieldsGlobalSizeDims[d] = threadParams->window.globalDimensions.size[d];
            threadParams->fieldsOffsetDims[d] = offset;
        }

        /* collect size information for each field to be written and define
         * field variables
         */
        log<picLog::INPUT_OUTPUT > ("ADIOS: (begin) collecting fields.");
        threadParams->adiosFieldVarIds.clear();
        if (threadParams->isCheckpoint)
        {
            ForEach<FileCheckpointFields, CollectFieldsSizes<bmpl::_1> > forEachCollectFieldsSizes;
            forEachCollectFieldsSizes(threadParams);
        }
        else
        {
            ForEach<FileOutputFields, CollectFieldsSizes<bmpl::_1> > forEachCollectFieldsSizes;
            forEachCollectFieldsSizes(threadParams);
        }
        log<picLog::INPUT_OUTPUT > ("ADIOS: ( end ) collecting fields.");

        /* collect size information for all attributes of all species and define
         * particle variables
         */
        threadParams->adiosParticleAttrVarIds.clear();
        threadParams->adiosSpeciesIndexVarIds.clear();
        log<picLog::INPUT_OUTPUT > ("ADIOS: (begin) counting particles.");
        if (threadParams->isCheckpoint)
        {
            ForEach<FileCheckpointParticles, ADIOSCountParticles<bmpl::_1> > adiosCountParticles;
            adiosCountParticles(threadParams, std::string());
        }
        else
        {
            ForEach<FileOutputParticles, ADIOSCountParticles<bmpl::_1> > adiosCountParticles;
            adiosCountParticles(threadParams, std::string());
        }
        log<picLog::INPUT_OUTPUT > ("ADIOS: ( end ) counting particles.");

        /* allocate buffer in MB according to our current group size */
        /* `1 + mem` minimum 1 MiB that we can write attributes on empty GPUs */
        size_t writeBuffer_in_MiB=1+threadParams->adiosGroupSize / 1024 / 1024;
        /* value `1.1` is the secure factor if we miss to count some small buffers*/
        size_t buffer_mem=static_cast<size_t>(1.1 * static_cast<double>(writeBuffer_in_MiB));
        ADIOS_CMD(adios_allocate_buffer(ADIOS_BUFFER_ALLOC_NOW,buffer_mem));
        threadParams->adiosBufferInitialized = true;

        /* open adios file. all variables need to be defined at this point */
        log<picLog::INPUT_OUTPUT > ("ADIOS: open file: %1%") % threadParams->fullFilename;
        ADIOS_CMD(adios_open(&(threadParams->adiosFileHandle), ADIOS_GROUP_NAME,
                threadParams->fullFilename.c_str(), "w", threadParams->adiosComm));

        if (threadParams->adiosFileHandle == ADIOS_INVALID_HANDLE)
            throw std::runtime_error("ADIOS: Failed to open file.");

        /* set adios group size (total size of all data to be written) */
        uint64_t adiosTotalSize;
        ADIOS_CMD(adios_group_size(threadParams->adiosFileHandle,
                threadParams->adiosGroupSize, &adiosTotalSize));

        writeMetaAttributes(threadParams);



        /* write fields */
        log<picLog::INPUT_OUTPUT > ("ADIOS: (begin) writing fields.");
        if (threadParams->isCheckpoint)
        {
            ForEach<FileCheckpointFields, GetFields<bmpl::_1> > forEachGetFields;
            forEachGetFields(threadParams);
        }
        else
        {
            ForEach<FileOutputFields, GetFields<bmpl::_1> > forEachGetFields;
            forEachGetFields(threadParams);
        }
        log<picLog::INPUT_OUTPUT > ("ADIOS: ( end ) writing fields.");

        /* print all particle species */
        log<picLog::INPUT_OUTPUT > ("ADIOS: (begin) writing particle species.");
        if (threadParams->isCheckpoint)
        {
            ForEach<FileCheckpointParticles, WriteSpecies<bmpl::_1> > writeSpecies;
            writeSpecies(threadParams, particleOffset);
        }
        else
        {
            ForEach<FileOutputParticles, WriteSpecies<bmpl::_1> > writeSpecies;
            writeSpecies(threadParams, particleOffset);
        }
        log<picLog::INPUT_OUTPUT > ("ADIOS: ( end ) writing particle species.");

        /* close adios file, most liekly the actual write point */
        log<picLog::INPUT_OUTPUT > ("ADIOS: closing file: %1%") % threadParams->fullFilename;
        ADIOS_CMD(adios_close(threadParams->adiosFileHandle));

        /*\todo: copied from adios example, we might not need this ? */
        MPI_CHECK(MPI_Barrier(threadParams->adiosComm));

        return NULL;
    }

    ThreadParams mThreadParams;

    MappingDesc *cellDescription;

    uint32_t notifyPeriod;
    int64_t lastCheckpoint;
    std::string filename;
    std::string checkpointFilename;
    std::string restartFilename;
    std::string checkpointDirectory;

    /* select MPI method, #OSTs and #aggregators */
    std::string mpiTransportParams;

    uint32_t restartChunkSize;

    DataSpace<simDim> mpi_pos;
    DataSpace<simDim> mpi_size;
};

} //namespace adios
} //namespace picongpu

