/**
 * Copyright 2014 Axel Huebl
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

#include "types.h"
#include "simulation_defines.hpp"
#include "basicOperations.hpp"

#include "dimensions/DataSpace.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "mappings/kernel/MappingDescription.hpp"
#include "simulationControl/MovingWindow.hpp"

/* just for testing */
#include "fields/FieldE.hpp"
#include "fields/FieldB.hpp"
#include "fields/FieldJ.hpp"
#include "fields/FieldTmp.hpp"

namespace picongpu
{
namespace cellwiseOperation
{
    using namespace PMacc;

    /** Kernel that calls OpFunctor on each cell of a field
     * 
     * \tparam T_OpFunctor<Field> not specialized (yet) for which field type
     * \tparam FieldBox field type
     * \tparam Mapping auto attached argument from __picKernelArea call
     */
    template<
        class T_OpFunctor,
        class T_ValFunctor,
        class FieldBox,
        class Mapping>
    __global__ void
    kernelCellwiseOperation( FieldBox field, T_OpFunctor opFunctor, T_ValFunctor valFunctor, const DataSpace<simDim> totalCellOffset,
        const uint32_t currentStep, Mapping mapper )
    {
        const DataSpace<simDim> block( mapper.getSuperCellIndex( DataSpace<simDim>( blockIdx ) ) );
        const DataSpace<simDim> blockCell = block * MappingDesc::SuperCellSize::getDataSpace();

        const DataSpace<simDim> threadIndex( threadIdx );
        
        opFunctor( field( blockCell + threadIndex ),
                   valFunctor( blockCell + threadIndex + totalCellOffset,
                               currentStep )
                 );
    }

    /** Call a functor on each cell of a field */
    class CellwiseOperation
    {
    private:
        typedef MappingDesc::SuperCellSize SuperCellSize;
        
        MappingDesc cellDescription;
        
    public:
        CellwiseOperation(MappingDesc cellDescription) : cellDescription(cellDescription)
        {
        }

        /* ...
         * 
         * \tparam AREA Where to compute on (CORE, BORDER, GUARD)
         * \tparam ValFunctor A Value-Producing functor for a given cell
         *                    in time and space
         * \tparam OpFunctor A manipulating functor like PMacc::nvidia::functors::add
         */
        template<uint32_t AREA, class T_Field, class T_OpFunctor, class T_ValFunctor>
        void
        exec( T_Field field, T_OpFunctor opFunctor, T_ValFunctor valFunctor, uint32_t currentStep, const bool enabled = true ) const
        {
            if( !enabled )
                return;

            /** offset due to slides AND due to being the n-th GPU */
            DataSpace<simDim> globalOffset(SubGrid<simDim>::getInstance().getSimulationBox().getGlobalOffset());
            VirtualWindow window = MovingWindow::getInstance().getVirtualWindow( currentStep );
            
            DataSpace<simDim> totalCellOffset(globalOffset);
            globalOffset.y() += window.slides * window.localFullSize.y();

            /* start kernel */
            __picKernelArea((kernelCellwiseOperation<T_OpFunctor>), cellDescription, AREA)
                    (SuperCellSize::getDataSpace())
                    (field->getDeviceDataBox(), opFunctor, valFunctor, totalCellOffset, currentStep);
        }
    };

} // namespace cellwiseOperation
} // namespace picongpu
