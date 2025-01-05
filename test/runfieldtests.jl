
using Test
using Logging

import PALEOboxes as PB
import PALEOmodel

#= function test_scalar_field(f::PB.Field)
    @test isnan(f.values[])
    f.values[] = 42.0

    # fa = PALEOmodel.get_array(f)
    # @test fa.dims == ()
    # @test fa.values[] == 42.0


    fr = PALEOmodel.FieldRecord(
        # (record_dim = PB.NamedDimension("records", -1), record_dim_coordinates=String[]),  # mock dataset to supply record_dim
        (record_dim = (name = "records",), ),  # mock dataset to supply record_dim.name
        f, 
        Dict{Symbol, Any}()
    )
    push!(fr, f)
    @test fr.records == [42.0]
    f.values[] = 43.0
    push!(fr, f)
    @test fr.records == [42.0, 43.0]
    @test fr[2].values == f.values

    fra = PALEOmodel.get_array(fr)

    @test fra.values == [42.0, 43.0]
    fra_dimensions = PB.get_dimensions(fra)
    @test length(fra_dimensions) == 1
    @test fra_dimensions[1].name == "records"

    frw = PALEOmodel.FieldRecord(
        (record_dim = (name = "records",), ),  # mock dataset to supply record_dim.name
        [42.0, 43.0], 
        PB.ScalarData, 
        (),
        PB.ScalarSpace, 
        nothing,
        Dict{Symbol, Any}(),
    )
    @test frw.records == [42.0, 43.0]
    @test frw[2].values == f.values

    

    return nothing
end
 =#

@testset "Fields" begin

    @testset "ScalarData, ScalarSpace" begin
        @info "ScalarData, ScalarSpace"

        f = PB.Field(PB.ScalarData, (), Float64, PB.ScalarSpace, nothing; allocatenans=true)
        
        @test isnan(f.values[])
        f.values[] = 42.0
    
        fr = PALEOmodel.FieldRecord(
            # (record_dim = PB.NamedDimension("records", -1), record_dim_coordinates=String[]),  # mock dataset to supply record_dim
            (record_dim = (name = "records",), ),  # mock dataset to supply record_dim.name
            f, 
            Dict{Symbol, Any}()
        )

        # FieldRecord with 1 record
        push!(fr, f)        
        @test PB.get_data(fr) == [42.0]
        @test PB.get_data(fr; records=1) == [42.0]
        fra = PALEOmodel.get_array(fr; squeeze_all_single_dims=false)
        @test size(fra.values) == (1, )
        @test fra.values == [42.0]  # Vector
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 1
        @test fra_dimensions[1].name == "records"

        fra = PALEOmodel.get_array(fr; squeeze_all_single_dims=true) # should squeeze out record dim
        @test size(fra.values) == ()
        @test fra.values == fill(42.0) # 0D Array
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 0
       

        # FieldRecord with 2 records
        f.values[] = 43.0
        push!(fr, f)
        @test PB.get_data(fr) == [42.0, 43.0]
        @test PB.get_data(fr; records=2) == [43.0]
        fra = PALEOmodel.get_array(fr; squeeze_all_single_dims=false)
        @test size(fra.values) == (2, ) #
        @test fra.values == [42.0, 43.0]
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 1
        @test fra_dimensions[1].name == "records"
    
    end

    

    function test_ScalarData_1cell(f::PB.Field)
        @test isnan(f.values[])
        f.values[] = 42.0
    
        fr = PALEOmodel.FieldRecord(
            # (record_dim = PB.NamedDimension("records", -1), record_dim_coordinates=String[]),  # mock dataset to supply record_dim
            (record_dim = (name = "records",), ),  # mock dataset to supply record_dim.name
            f, 
            Dict{Symbol, Any}()
        )

        # FieldRecord with 1 record
        push!(fr, f)        
        @test PB.get_data(fr) == [42.0]
        @test PB.get_data(fr; records=1) == [42.0]
        fra = PALEOmodel.get_array(fr; squeeze_all_single_dims=false)
        @test size(fra.values) == (1, 1)
        @test fra.values == reshape([42.0], 1, 1) # 1×1 Matrix{Float64}
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 2
        @test fra_dimensions[1].name == "cells"
        @test fra_dimensions[2].name == "records"

        # FieldArray, should squeeze out cells and record dim
        fra = PALEOmodel.get_array(fr; squeeze_all_single_dims=true) 
        @test size(fra.values) == ()
        @test fra.values == fill(42.0) # 0D Array
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 0

        # FieldRecord with 2 records
        f.values[] = 43.0
        push!(fr, f)
        @test PB.get_data(fr) == [42.0, 43.0]
        @test PB.get_data(fr; records=2) == [43.0]
        fra = PALEOmodel.get_array(fr; squeeze_all_single_dims=false)
        @test size(fra.values) == (1, 2)
        @test fra.values == [42.0 43.0] # 1×2 Matrix{Float64}
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 2
        @test fra_dimensions[1].name == "cells"
        @test fra_dimensions[2].name == "records"

        # FieldArray should squeeze out cells  dim
        fra = PALEOmodel.get_array(fr; squeeze_all_single_dims=true) 
        @test size(fra.values) == (2, )
        @test fra.values == [42.0, 43.0] # Vector
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 1
        @test fra_dimensions[1].name == "records"

        # FieldArray should be zero dim
        fra = PALEOmodel.get_array(fr, (cell=1, records=1)) 
        @test size(fra.values) == ()
        @test fra.values == fill(42.0) # 0D array
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 0

    end


    @testset "ScalarData, CellSpace, no grid" begin
        @info "ScalarData, CellSpace, no grid" 

        f = PB.Field(PB.ScalarData, (), Float64, PB.CellSpace, nothing; allocatenans=true)
        
        test_ScalarData_1cell(f)

    end

    @testset "ScalarData, CellSpace, grid with 1 cell" begin
        @info "ScalarData, CellSpace, grid with 1 cell"
        # should behave identically to no grid case

        g = PB.Grids.UnstructuredVectorGrid(ncells=1)

        f = PB.Field(PB.ScalarData, (), Float64, PB.CellSpace, g; allocatenans=true)

        test_ScalarData_1cell(f)
    end

    @testset "ScalarData, CellSpace, UnstructuredColumnGrid" begin
        @info "ScalarData, CellSpace, UnstructuredColumnGrid"

        g = PB.Grids.UnstructuredColumnGrid(ncells=5, Icolumns=[collect(1:5)])

        f = PB.Field(PB.ScalarData, (), Float64, PB.CellSpace, g; allocatenans=true)
        
        f.values .= 42.0
        fr = PALEOmodel.FieldRecord(
            (record_dim = (name = "records",), ),  # mock dataset to supply record_dim.name
            f, 
            Dict{Symbol, Any}(), 
        )
        push!(fr, f)
        f.values .= 43.0
        push!(fr, f)

        @test PB.get_data(fr) == [fill(42.0, 5), fill(43.0, 5)] # Vector of Vectors

        @test fr[1] isa PB.Field
        @test fr[1].values == fill(42.0, 5)

        fa = PALEOmodel.get_array(fr, (column=1, ))
        fa_dimensions = PB.get_dimensions(fa)
        @test length(fa_dimensions) == 2
        @test fa_dimensions[1].name == "cells"
        @test fa_dimensions[2].name == "records"
        @test size(fa.values) == (5, 2)
        @test fa.values[:, 1] == fill(42.0, 5)
        @test fa.values[:, 2] == fill(43.0, 5)

        fa_d = PALEOmodel.get_array(fr, column=1) # deprecated syntax
        @test fa.values == fa_d.values
    end
  
    @testset "ArrayScalarData, ScalarSpace" begin
        @info "ArrayScalarData, ScalarSpace"

        f = PB.Field(PB.ArrayScalarData, (PB.NamedDimension("test", 2), ), Float64, PB.ScalarSpace, nothing; allocatenans=true)

        @test isnan.(f.values) == [true, true]
        f.values .= [42.0, 43.0]

        fr = PALEOmodel.FieldRecord(
            (record_dim = (name = "records",), ),  # mock dataset to supply record_dim.name
            f, 
            Dict{Symbol, Any}(),
        )
        push!(fr, f)
        @test PB.get_data(fr) == [[42.0, 43.0]] # Vector of Vectors

        # FieldArray from FieldRecord - records dim squeezed out
        fra = PALEOmodel.get_array(fr)
        @test size(fra.values) == (2, )
        @test fra.values == [42.0, 43.0]
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 1
        @test fra_dimensions[1].name == "test"
        # FieldArray from FieldRecord - records dim not squeezed out
        fra = PALEOmodel.get_array(fr; squeeze_all_single_dims=false)
        @test size(fra.values) == (2, 1)
        @test fra.values == reshape([42.0, 43.0], 2, 1) # 2×1 Matrix{Float64}
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 2
        @test fra_dimensions[1].name == "test"
        @test fra_dimensions[2].name == "records"

        f.values .= [44.0, 45.0]
        push!(fr, f)
        @test PB.get_data(fr) == [[42.0, 43.0], [44.0, 45.0]] # Vector of Vectors
        @test PB.get_data(fr; records=2) == [44.0, 45.0] # Vector
        @test fr[2].values == f.values


        # FieldArray from FieldRecord
        fra = PALEOmodel.get_array(fr)
        @test fra.values == [42.0 44.0; 43.0 45.0]
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 2
        @test fra_dimensions[1].name == "test"
        @test fra_dimensions[2].name == "records"
    end

    function test_ArrayScalarData_1cell(f)
        @test size(f.values) == (1, 2) # Matrix size(1, 2) 
        @test isnan.(f.values) == [true true] # 
        f.values .= [42.0 43.0]

        fr = PALEOmodel.FieldRecord(
            (record_dim = (name = "records",), ),  # mock dataset to supply record_dim.name
            f, 
            Dict{Symbol, Any}(),
        )
        push!(fr, f)
        @test PB.get_data(fr) == [[42.0 43.0]]
        f.values .= [44.0 45.0]
        # FieldArray from FieldRecord, records dimension squeezed out
        fra = PALEOmodel.get_array(fr)
        @test size(fra.values) == (2,)
        @test fra.values == [42.0, 43.0]
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 1
        @test fra_dimensions[1].name == "test"
        @test fra_dimensions[1].size == 2


        push!(fr, f)
        @test PB.get_data(fr) == [[42.0 43.0], [44.0 45.0]]
        @test PB.get_data(fr; records=2) == [44.0 45.0]
        @test fr[2].values == f.values


        # FieldArray from FieldRecord
        fra = PALEOmodel.get_array(fr)
        @test fra.values == [42.0 44.0; 43.0 45.0]
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 2
        @test fra_dimensions[1].name == "test"
        @test fra_dimensions[1].size == 2
        @test fra_dimensions[2].name == "records"
        @test fra_dimensions[2].size == 2

        # FieldArray should be 1 D
        fra = PALEOmodel.get_array(fr, (cell=1, records=1)) 
        @test size(fra.values) == (2,)
        @test fra.values == [42.0, 43.0]
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 1
        @test fra_dimensions[1].name == "test"
        @test fra_dimensions[1].size == 2

        # FieldArray should be 1 D
        fra = PALEOmodel.get_array(fr, (cell=1, test_isel=2)) 
        @test size(fra.values) == (2,)
        @test fra.values == [43.0, 45.0]
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 1
        @test fra_dimensions[1].name == "records"
        @test fra_dimensions[1].size == 2
        
        # FieldArray should be zero dim
        fra = PALEOmodel.get_array(fr, (cell=1, records=1, test_isel=2)) 
        @test size(fra.values) == ()
        @test fra.values == fill(43.0) # 0D array
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 0
    end

    @testset "ArrayScalarData, CellSpace, no grid" begin
        @info "ArrayScalarData, CellSpace, no grid"

        # NB: Field.values here is a (1, 2) Array, not a (2,) Vector
        f = PB.Field(PB.ArrayScalarData, (PB.NamedDimension("test", 2), ), Float64, PB.CellSpace, nothing; allocatenans=true)

        test_ArrayScalarData_1cell(f)
        
    end

    @testset "ArrayScalarData, CellSpace, grid with 1 cell" begin
        @info "ArrayScalarData, CellSpace, grid with 1 cell"
        # should behave identically to no grid case

        g = PB.Grids.UnstructuredVectorGrid(ncells=1)

        # NB: Field.values here is a (1, 2) Array, not a (2,) Vector
        f = PB.Field(PB.ArrayScalarData, (PB.NamedDimension("test", 2), ), Float64, PB.CellSpace, g; allocatenans=true)

        test_ArrayScalarData_1cell(f)

    end

    @testset "ArrayScalarData, CellSpace, UnstructuredColumnGrid" begin
        @info "ArrayScalarData, CellSpace, UnstructuredColumnGrid"

        g = PB.Grids.UnstructuredColumnGrid(ncells=5, Icolumns=[collect(1:5)])

        f = PB.Field(PB.ArrayScalarData, (PB.NamedDimension("test", 2), ), Float64, PB.CellSpace, g; allocatenans=true)

        @test size(f.values) == (5, 2)

        f.values .= 42.0
        fr = PALEOmodel.FieldRecord(
            (record_dim = (name = "records",), ),  # mock dataset to supply record_dim.name
            f, 
            Dict{Symbol, Any}(),
        )
        push!(fr, f)
        f.values .= 43.0
        push!(fr, f)

        @test PB.get_data(fr) == [fill(42.0, 5, 2), fill(43.0, 5, 2)]

        @test fr[1].values == fill(42.0, 5, 2)

        # FieldArray from FieldRecord
        fra = PALEOmodel.get_array(fr, (column=1))
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 3
        @test fra_dimensions[1].name == "cells"
        @test fra_dimensions[2].name == "test"
        @test fra_dimensions[3].name == "records"
        @test size(fra.values) == (5, 2, 2)
        @test fra.values[:, :, 1] == fill(42.0, 5, 2)
        @test fra.values[:, :, 2] == fill(43.0, 5, 2)

        fra = PALEOmodel.get_array(fr, (column=1, cell=1))
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 2
        @test fra_dimensions[1].name == "test"
        @test fra_dimensions[2].name == "records"
        @test size(fra.values) == (2, 2)
        @test fra.values[:, 1] == fill(42.0, 2)
        @test fra.values[:, 2] == fill(43.0, 2)
    end


    @testset "copy FieldRecord" begin
        @info "copy FieldRecord"

        g = PB.Grids.UnstructuredColumnGrid(ncells=5, Icolumns=[collect(1:5)])

        f = PB.Field(PB.ScalarData, (), Float64, PB.CellSpace, g; allocatenans=true)
        
        f.values .= 42.0
        fr = PALEOmodel.FieldRecord(
            (record_dim = (name = "records",), ),  # mock dataset to supply record_dim.name
            f, 
            Dict{Symbol, Any}(), 
        )
        push!(fr, f)
        f.values .= 43.0
        push!(fr, f)

        @test fr.records == [fill(42.0, 5), fill(43.0, 5)]

        fr_copy = copy(fr)
        @test fr_copy.records == [fill(42.0, 5), fill(43.0, 5)]
        push!(fr_copy, f)
        @test fr_copy.records == [fill(42.0, 5), fill(43.0, 5),  fill(43.0, 5)]
        @test length(fr_copy) == 3
        @test PB.get_dimension(fr_copy, "records") == PB.NamedDimension("records", 3)

        @test fr.records == [fill(42.0, 5), fill(43.0, 5)]
        @test PB.get_dimension(fr, "records") == PB.NamedDimension("records", 2)

    end
end
