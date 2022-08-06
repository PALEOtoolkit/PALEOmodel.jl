using Test
using Logging

import DataFrames

import PALEOboxes as PB
import PALEOmodel

function create_test_model_output(nrecords::Int)
 
    model = PB.create_model_from_config(
        joinpath(@__DIR__, "configreservoirs.yaml"),
        "model1",
    )

    intial_state, modeldata = PALEOmodel.initialize!(model)

    all_vars = PB.VariableAggregatorNamed(modeldata)
    println(all_vars)
    
    output = PALEOmodel.OutputWriters.OutputMemory()
    PALEOmodel.OutputWriters.initialize!(output, model, modeldata, nrecords)

    return model, modeldata, all_vars, output
end

@testset "OutputWriters" begin

@testset "ModelOutput" begin
    # Test creation from model output

    model, modeldata, all_vars, output =  create_test_model_output(2)
    all_values = all_vars.values

    all_values.global.O .= [2e19]
    PALEOmodel.OutputWriters.add_record!(output, model, modeldata, 0.0)

    all_values.global.O .= [4e19]
    PALEOmodel.OutputWriters.add_record!(output, model, modeldata, 1.0)

    O_array = PALEOmodel.get_array(output, "global.O")
    @test O_array.values == [2e19, 4e19]

end

@testset "ModelOutputShort" begin
    # Test creation from model output, partially filling output (4 records capacity, only 2 records will be written)

    model, modeldata, all_vars, output =  create_test_model_output(4) 
    all_values = all_vars.values

    all_values.global.O .= [2e19]
    PALEOmodel.OutputWriters.add_record!(output, model, modeldata, 0.0)

    all_values.global.O .= [4e19]
    PALEOmodel.OutputWriters.add_record!(output, model, modeldata, 0.0)

    O_array = PALEOmodel.get_array(output, "global.O")
    @test O_array.values == [2e19, 4e19]

end

@testset "SaveLoad" begin

    model, modeldata, all_vars, output =  create_test_model_output(2)
    all_values = all_vars.values

    all_values.global.O .= [2e19]
    PALEOmodel.OutputWriters.add_record!(output, model, modeldata, 0.0)
    all_values.global.O .= [4e19]
    PALEOmodel.OutputWriters.add_record!(output, model, modeldata, 0.0)

    tmpfile = tempname(; cleanup=true) 
    PALEOmodel.OutputWriters.save_jld2(output, tmpfile)

    load_output = PALEOmodel.OutputWriters.load_jld2!(PALEOmodel.OutputWriters.OutputMemory(), tmpfile)

    O_array = PALEOmodel.get_array(load_output, "global.O")
    @test O_array.values == [2e19, 4e19]

end

@testset "DataFrameCreate" begin
    # Test creation from a DataFrame        

    tmodel = collect(0.0:0.1:10.0)
    test = ones(length(tmodel))

    glb = DataFrames.DataFrame(tmodel=tmodel, test=test)
 
    test_output_global = PALEOmodel.OutputWriters.OutputMemoryDomain("global", glb)
    test_output = PALEOmodel.OutputWriters.OutputMemory([test_output_global])

    test_array = PALEOmodel.get_array(test_output, "global.test")

    @test test_array.values == test
end


end


