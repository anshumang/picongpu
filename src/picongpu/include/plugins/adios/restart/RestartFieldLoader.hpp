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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <adios.h>
#include <adios_read.h>
#include <adios_error.h>

#include <string>
#include <sstream>

#include "types.h"
#include "simulation_defines.hpp"
#include "plugins/adios/ADIOSWriter.def"

#include "particles/frame_types.hpp"
#include "dataManagement/DataConnector.hpp"
#include "dimensions/DataSpace.hpp"
#include "dimensions/GridLayout.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldB.hpp"
#include "simulationControl/MovingWindow.hpp"

namespace picongpu
{

namespace adios
{

/**
 * Helper class for ADIOS plugin to load fields from parallel ADIOS BP files.
 */
class RestartFieldLoader
{
public:
    template<class Data>
    static void loadField(Data& field, const uint32_t numComponents, std::string objectName, ThreadParams *params)
    {
        log<picLog::INPUT_OUTPUT > ("Begin loading field '%1%'") % objectName;

        const std::string name_lookup_tpl[] = {"x", "y", "z", "w"};
        const DataSpace<simDim> field_guard = field.getGridLayout().getGuard();

        const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(params->currentStep);
        const PMacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();

        field.getHostBuffer().setValue(float3_X(0.));

        //const std::string name_lookup[] = {"x", "y", "z"};

        /* globalSlideOffset due to gpu slides between origin at time step 0
         * and origin at current time step
         * ATTENTION: splash offset are globalSlideOffset + picongpu offsets
         */
        DataSpace<simDim> globalSlideOffset;
        globalSlideOffset.y() = numSlides * localDomain.size.y();

        DataSpace<simDim> domain_offset;
        for (uint32_t d = 0; d < simDim; ++d)
            domain_offset[d] = localDomain.offset[d] + globalSlideOffset[d];

        if (Environment<simDim>::get().GridController().getPosition().y() == 0)
            domain_offset[1] += params->window.globalDimensions.offset.y();

        DataSpace<simDim> local_domain_size;
        for (uint32_t d = 0; d < simDim; ++d)
            local_domain_size[d] = params->window.localDimensions.size[d];

        PMACC_AUTO(destBox, field.getHostBuffer().getDataBox());
        for (uint32_t n = 0; n < numComponents; ++n)
        {
            // Read the subdomain which belongs to our mpi position.
            // The total grid size must match the grid size of the stored data.
            log<picLog::INPUT_OUTPUT > ("ADIOS: Read from domain: offset=%1% size=%2%") %
                domain_offset % local_domain_size;

            std::stringstream datasetName;
            datasetName << params->adiosBasePath << ADIOS_PATH_FIELDS << objectName;
            if (numComponents > 1)
                datasetName << "/" << name_lookup_tpl[n];

            log<picLog::INPUT_OUTPUT > ("ADIOS: Read from field '%1%'") %
                datasetName.str();

            ADIOS_VARINFO* varInfo = adios_inq_var( params->fp, datasetName.str().c_str() );
            uint64_t start[varInfo->ndim];
            uint64_t count[varInfo->ndim];
            for(int d = 0; d < varInfo->ndim; ++d)
            {
                start[d] = domain_offset[d];
                count[d] = local_domain_size[d];
            }

            ADIOS_SELECTION* fSel = adios_selection_boundingbox( varInfo->ndim, start, count );

            /* specify what we want to read, but start reading at below at
             * `adios_perform_reads` */
            log<picLog::INPUT_OUTPUT > ("ADIOS: Allocate %1% elements") %
                local_domain_size.productOfComponents();

            /// \todo float_X should be some kind of gridBuffer's GetComponentsType<ValueType>::type
            float_X* field_container = new float_X[local_domain_size.productOfComponents()];
            ADIOS_CMD(adios_schedule_read( params->fp, fSel, datasetName.str().c_str(), 0, 1, (void*)field_container ));

            /* start a blocking read of all scheduled variables */
            ADIOS_CMD(adios_perform_reads( params->fp, 1 ));

            int elementCount = params->window.localDimensions.size.productOfComponents();

            for (int linearId = 0; linearId < elementCount; ++linearId)
            {
                /* calculate index inside the moving window domain which is located on the local grid*/
                DataSpace<simDim> destIdx = DataSpaceOperations<simDim>::map(params->window.localDimensions.size, linearId);
                /* jump over guard and local sliding window offset*/
                destIdx += field_guard + params->localWindowToDomainOffset;

                destBox(destIdx)[n] = field_container[linearId];
            }

            __deleteArray(field_container);
            adios_selection_delete(fSel);
            adios_free_varinfo(varInfo);
        }

        field.hostToDevice();

        __getTransactionEvent().waitForFinished();

        log<picLog::INPUT_OUTPUT > ("ADIOS: Read from domain: offset=%1% size=%2%") %
            domain_offset % local_domain_size;
        log<picLog::INPUT_OUTPUT > ("ADIOS: Finished loading field '%1%'") % objectName;
    }

    template<class Data>
    static void cloneField(Data& fieldDest, Data& fieldSrc, std::string objectName)
    {
        log<picLog::INPUT_OUTPUT > ("ADIOS: Begin cloning field '%1%'") % objectName;
        DataSpace<simDim> field_grid = fieldDest.getGridLayout().getDataSpace();

        size_t elements = field_grid.productOfComponents();
        float3_X *ptrDest = fieldDest.getHostBuffer().getDataBox().getPointer();
        float3_X *ptrSrc = fieldSrc.getHostBuffer().getDataBox().getPointer();

        for (size_t k = 0; k < elements; ++k)
        {
            ptrDest[k] = ptrSrc[k];
        }

        fieldDest.hostToDevice();

        __getTransactionEvent().waitForFinished();

        log<picLog::INPUT_OUTPUT > ("ADIOS: Finished cloning field '%1%'") % objectName;
    }
};

/**
 * Hepler class for ADIOSWriter (forEach operator) to load a field from ADIOS
 *
 * @tparam FieldType field class to load
 */
template< typename FieldType >
struct LoadFields
{
public:

    HDINLINE void operator()(ThreadParams* params)
    {
#ifndef __CUDA_ARCH__
        DataConnector &dc = Environment<>::get().DataConnector();
        ThreadParams *tp = params;

        /* load field without copying data to host */
        FieldType* field = &(dc.getData<FieldType > (FieldType::getName(), true));

        /* load from ADIOS */
        RestartFieldLoader::loadField(
                field->getGridBuffer(),
                (uint32_t)FieldType::numComponents,
                FieldType::getName(),
                tp);

        dc.releaseData(FieldType::getName());
#endif
    }

};

using namespace PMacc;

} /* namespace adios */
} /* namespace picongpu */
