# ************************************************************************
# FAUST Architecture File
# Copyright (C) 2022 GRAME, Centre National de Creation Musicale
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
    my_dsp.instanceConstants(SAMPLERATE)
    my_dsp.instanceResetUserInterface()

    print("getNumInputs: ", my_dsp.getNumInputs())
    print("getNumOutputs: ", my_dsp.getNumOutputs())

    import json

    json_str = my_dsp.getJSON()
    json_obj = json.loads(json_str)
    print('json_obj: ', json_obj)
    
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

    duration = 1.

    inputs = torch.zeros(my_dsp.getNumInputs(), int(SAMPLERATE*duration), dtype=dtype, device=device)
    outputs = torch.zeros(my_dsp.getNumOutputs(), int(SAMPLERATE*duration), dtype=dtype, device=device)
    count = outputs.shape[1]
    print('count: ', count)
    output = my_dsp(inputs, outputs, count)

    # todo: assert the shape of the output with getNumOutputs()
    assert(output.shape[0] == outputs.shape[0])
    
    # todo: display the outputs

def main():
    test()
    pass

if __name__ == '__main__':
    main()