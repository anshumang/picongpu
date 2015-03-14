/**
 * Copyright 2014-2015 Anshuman Goswami
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

#include "plugins/protobuf/PROTOBUFWriter.def"

namespace picongpu
{

namespace protobuf
{

using namespace PMacc;

/**
 * Writes simulation data to protocol buffers.
 * Implements the ILightweightPlugin interface.
 */
class PROTOBUFWriter : public ILightweightPlugin
{
private:

    template<typename UnitType>
    static std::vector<double> createUnit(UnitType unit, uint32_t numComponents)
    {
        std::vector<double> tmp(numComponents);
        for (uint i = 0; i < numComponents; ++i)
            tmp[i] = unit[i];
        return tmp;
    }

    /**
     * Collect field sizes to set protobuf group size.
     */
    template< typename T >
    struct CollectFieldsSizes
    {
    public:
        typedef typename T::ValueType ValueType;
        typedef typename T::UnitValueType UnitType;
        typedef typename GetComponentsType<ValueType>::type ComponentType;

        static std::vector<double> getUnit()
        {
            UnitType unit = T::getUnit();
            return createUnit(unit, T::numComponents);
        }

        HDINLINE void operator()(ThreadParams* params)
        {
#ifndef __CUDA_ARCH__
            const uint32_t components = T::numComponents;

            // protobuf buffer size for this dataset (all components)
            uint64_t localGroupSize =
                    params->window.localDimensions.size.productOfComponents() *
                    sizeof(ComponentType) *
                    components;

             //defineFieldVar for protobuf
#endif
        }
    };

    /**
     * Collect field sizes to set protobuf group size.
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
        typedef typename FieldTmp::UnitValueType UnitType;
        typedef typename GetComponentsType<ValueType>::type ComponentType;

        /** Create a name for the protobuf identifier.
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
            UnitType unit = FieldTmp::getUnit<Solver>();
            const uint32_t components = GetNComponents<ValueType>::value;
            return createUnit(unit, components);
        }

        HINLINE void operator_impl(ThreadParams* params)
        {
            const uint32_t components = GetNComponents<ValueType>::value;

            // adios buffer size for this dataset (all components)
            uint64_t localGroupSize =
                    params->window.localDimensions.size.productOfComponents() *
                    sizeof(ComponentType) *
                    components;

             //defineFieldVar for protobuf
        }

    };
    /**
     * Write calculated fields to a protobuf buffer.
     */
    template< typename T >
    struct GetFields
    {
    private:
        typedef typename T::ValueType ValueType;
        typedef typename GetComponentsType<ValueType>::type ComponentType;

    public:

        HDINLINE void operator()(ThreadParams* params)
        {
#ifndef __CUDA_ARCH__
            DataConnector &dc = Environment<simDim>::get().DataConnector();

            T* field = &(dc.getData<T > (T::getName()));
            params->gridLayout = field->getGridLayout();

            writeField(params,
                       sizeof(ComponentType),
                       GetNComponents<ValueType>::value,
                       T::getName(),
                       field->getHostDataBox().getPointer());

            dc.releaseData(T::getName());
#endif
        }

    };

    /** Calculate FieldTmp with given solver and particle species
     * and write them to protobuf.
     *
     * FieldTmp is calculated on device and than dumped to protobuf.
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

        /** Create a name for the protobuf identifier.
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

            params->gridLayout = fieldTmp->getGridLayout();
            /*write data to PROTOBUF buffer*/
            writeField(params,
                       sizeof(ComponentType),
                       components,
                       getName(),
                       fieldTmp->getHostDataBox().getPointer());

            dc.releaseData(FieldTmp::getName());

        }

    };


public:

    PROTOBUFWriter() :
    notifyPeriod(0)
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    virtual ~PROTOBUFWriter()
    {

    }

    void pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ("protobuf.period", po::value<uint32_t > (&notifyPeriod)->default_value(0),
             "enable PROTOBUF IO [for each n-th step]");
    }

    std::string pluginGetName() const
    {
        return "PROTOBUFWriter";
    }

    __host__ void notify(uint32_t currentStep)
    {
        notificationReceived(currentStep, false);
    }

private:

    /**
     * Notification for dump or checkpoint received
     *
     * @param currentStep current simulation step
     * @param isCheckpoint checkpoint notification
     */
    void notificationReceived(uint32_t currentStep, bool isCheckpoint)
    {
        writeProtobuf((void*) &mThreadParams);
    }

    void pluginLoad()
    {
        if (notifyPeriod > 0)
        {
            Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
        }

        loaded = true;
    }

    void pluginUnload()
    {

    }

    static void writeField(ThreadParams *params, const uint32_t sizePtrType,
                           const uint32_t nComponents, const std::string name,
                           void *ptr)
    {
        log<picLog::INPUT_OUTPUT > ("PROTOBUF: write field: %1% %2% %3%") %
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

            /* copy strided data from source to temporary buffer
             *
             * \todo use d1Access as in `include/plugins/hdf5/writer/Field.hpp`
             */
            const int maxZ = simDim == DIM3 ? field_no_guard[2] : 1;
            const int guardZ = simDim == DIM3 ? field_guard[2] : 0;
            for (int z = 0; z < maxZ; ++z)
            {
                for (int y = 0; y < field_no_guard[1]; ++y)
                {
                    const size_t base_index_src =
                                (z + guardZ) * plane_full_size +
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
   
            //PROTOBUF write goes here

        }
    }

    /**
     * Write meta attributes
     *
     * @param threadParams parameters
     */
    static void writeMetaAttributes(ThreadParams *threadParams)
    {
        log<picLog::INPUT_OUTPUT > ("PROTOBUF: (begin) write meta attributes.");

        /* write number of slides to timestep in adios file */
        uint32_t slides = MovingWindow::getInstance().getSlideCounter(threadParams->currentStep);

        //write the meta attributes to PROTOBUF here

        log<picLog::INPUT_OUTPUT > ("PROTOBUF: ( end ) wite meta attributes.");
    }

    static void *writeProtobuf(void *p_args)
    {
        ThreadParams *threadParams = (ThreadParams*) (p_args);

        /* y direction can be negative for first gpu */
        DataSpace<simDim> particleOffset(Environment<simDim>::get().SubGrid().getLocalDomain().offset);
        particleOffset.y() -= threadParams->window.globalDimensions.offset.y();

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
        log<picLog::INPUT_OUTPUT > ("PROTOBUF: (begin) collecting fields.");
        //Use PROTOBUF only for; output not checkpoint
	ForEach<FileOutputFields, CollectFieldsSizes<bmpl::_1> > forEachCollectFieldsSizes;
	forEachCollectFieldsSizes(threadParams);
        log<picLog::INPUT_OUTPUT > ("PROTOBUF: ( end ) collecting fields.");

        /* attributes written here are pure meta data */
        writeMetaAttributes(threadParams);

        /* write fields */
        log<picLog::INPUT_OUTPUT > ("PROTOBUF: (begin) writing fields.");
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

        return NULL;
    }

    ThreadParams mThreadParams;

    uint32_t notifyPeriod;
};

} //namespace protobuf
} //namespace picongpu


