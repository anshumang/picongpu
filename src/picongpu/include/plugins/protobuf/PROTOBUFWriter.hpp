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

    static void *writeProtobuf(void *p_args)
    {

        return NULL;
    }

    uint32_t notifyPeriod;
};

} //namespace protobuf
} //namespace picongpu


