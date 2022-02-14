# ************************************************************************
# FAUST Architecture File
# Copyright (C) 2021 GRAME, Centre National de Creation Musicale
# ---------------------------------------------------------------------

# This is sample code. This file is provided as an example of minimal
# FAUST architecture file. Redistribution and use in source and binary
# forms, with or without modification, in part or in full are permitted.
# In particular you can create a derived work of this FAUST architecture
# and distribute that work under terms of your choice.

# This sample code is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ************************************************************************

# Generated code
<<includeIntrinsic>>
<<includeclass>>

SAMPLERATE = 44100


def test():

    # Init DSP
    my_dsp = mydsp()
    my_dsp.classInit(SAMPLERATE)

    print("getNumInputs: ", my_dsp.getNumInputs())
    print("getNumOutputs: ", my_dsp.getNumOutputs())
    
    # # Create a MapUI controller
    # map_ui = MapUI(my_dsp)
    # my_dsp.buildUserInterface(map_ui)

    # # Print all zones
    # print("Path/UIZone dictionary: ", getZoneMap(map_ui))
     
    # # Possibly change control values
    # - using simple labels (end of path):
    # map_ui.setParamValue("freq", 500.0)
    # map_ui.setParamValue("/volume", -10.0)
    # - or using complete path:
    # map_ui.setParamValue("/Oscillator/freq", 500.0)
    # map_ui.setParamValue("/Oscillator/volume", -10.0)

    inputs = torch.zeros(my_dsp.getNumInputs(), dtype=dtype)  # FAUSTFLOAT?
    output = my_dsp(inputs)

    # todo: assert the shape of the output with getNumOutputs()
    
    # todo: display the outputs

def main():
    pass

if __name__ == '__main__':
    main()