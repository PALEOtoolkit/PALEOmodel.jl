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

@testset "SaveLoad_jld2" begin

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

@testset "SaveLoad_netcdf" begin

    model, modeldata, all_vars, output =  create_test_model_output(2)
    all_values = all_vars.values

    all_values.global.O .= [2e19]
    PALEOmodel.OutputWriters.add_record!(output, model, modeldata, 0.0)
    all_values.global.O .= [4e19]
    PALEOmodel.OutputWriters.add_record!(output, model, modeldata, 0.0)

    tmpfile = tempname(; cleanup=true)

    output.user_data["testString"] = "hello"
    output.user_data["testInt64"] = 42
    output.user_data["testFloat64"] = 42.0
    output.user_data["testVecString"] = ["hello", "world"]
    output.user_data["testVecInt64"] = [42, 43]
    output.user_data["testVecFloat64_1"] = [42.0]
    output.user_data["testVecFloat64"] = [42.0, 43.0]

    PALEOmodel.OutputWriters.save_netcdf(output, tmpfile)

    load_output = PALEOmodel.OutputWriters.load_netcdf!(PALEOmodel.OutputWriters.OutputMemory(), tmpfile)

    O_array = PALEOmodel.get_array(load_output, "global.O")
    @test O_array.values == [2e19, 4e19]

    function test_user_key_type_value(k, v)
        lv = load_output.user_data[k]
        @test typeof(lv) == typeof(v)
        @test lv == v
    end
      
    test_user_key_type_value("testString", "hello")
    test_user_key_type_value("testInt64", 42)
    test_user_key_type_value("testFloat64", 42.0)
    test_user_key_type_value("testVecString", ["hello", "world"])
    test_user_key_type_value("testVecInt64", [42, 43])
    @test_broken load_output.user_data["testVecFloat64_1"] == [42.0] # returned as a scalar
    test_user_key_type_value("testVecFloat64", [42.0, 43.0])

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


