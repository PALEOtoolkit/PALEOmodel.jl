
using Test
using Logging

import PALEOboxes as PB
import PALEOmodel

function test_scalar_field(f::PB.Field)
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


@testset "Fields" begin

    @testset "ScalarData, ScalarSpace" begin

        f = PB.Field(PB.ScalarData, (), Float64, PB.ScalarSpace, nothing; allocatenans=true)
        
        test_scalar_field(f)
    end

    @testset "ScalarData, CellSpace, no grid" begin
        # check that a CellSpace Field with no grid behaves as a ScalarSpace Field

        f = PB.Field(PB.ScalarData, (), Float64, PB.CellSpace, nothing; allocatenans=true)
        
        test_scalar_field(f)
    end

    @testset "ScalarData, CellSpace, UnstructuredColumnGrid" begin
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
        f = PB.Field(PB.ArrayScalarData, (PB.NamedDimension("test", 2), ), Float64, PB.ScalarSpace, nothing; allocatenans=true)

        @test isnan.(f.values) == [true, true]
        f.values .= [42.0, 43.0]

        # fa = PALEOmodel.get_array(f)
        # @test length(fa.dims) == 1
        # @test fa.dims[1].name == "test"
        # @test fa.dims[1].size == 2

        # @test fa.values == [42.0, 43.0]


        fr = PALEOmodel.FieldRecord(
            (record_dim = (name = "records",), ),  # mock dataset to supply record_dim.name
            f, 
            Dict{Symbol, Any}(),
        )
        push!(fr, f)
        @test fr.records == [[42.0, 43.0]]
        f.values .= [44.0, 45.0]
        push!(fr, f)
        @test fr.records == [[42.0, 43.0], [44.0, 45.0]]
        @test fr[2].values == f.values


        # FieldArray from Field
        # fa = PALEOmodel.get_array(f)
        # @test fa.values == [44.0, 45.0]
        # @test length(fa.dims) == 1
        # @test fa.dims[1].name == "test"

        # FieldArray from FieldRecord
        fra = PALEOmodel.get_array(fr)
        @test fra.values == [42.0 44.0; 43.0 45.0]
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 2
        @test fra_dimensions[1].name == "test"
        @test fra_dimensions[2].name == "records"
    end

    @testset "ArrayScalarData, CellSpace, no grid" begin
        # TODO this is possibly inconsistent with (ScalarData, CellSpace, no grid),
        # as Field.values here is a (1, 2) Array, not a (2,) Vector
        f = PB.Field(PB.ArrayScalarData, (PB.NamedDimension("test", 2), ), Float64, PB.CellSpace, nothing; allocatenans=true)

        @test_broken size(f.values) == (2, ) # TODO should be a Vector ?
        @test size(f.values) == (1, 2)
        @test isnan.(f.values) == [true true] # TODO 1x2 Array or Vector ?
        f.values .= [42.0 43.0]  # TODO

        fr = PALEOmodel.FieldRecord(
            (record_dim = (name = "records",), ),  # mock dataset to supply record_dim.name
            f, 
            Dict{Symbol, Any}(),
        )
        push!(fr, f)
        @test fr.records == [[42.0 43.0]]
        f.values .= [44.0 45.0]
        push!(fr, f)
        @test fr.records == [[42.0 43.0], [44.0 45.0]]
        @test fr[2].values == f.values

        # FieldArray from Field
        # fa = PALEOmodel.get_array(f)
        # @test fa.values == [44.0, 45.0]
        # @test length(fa.dims) == 1
        # @test fa.dims[1].name == "test"
        # @test fa.dims[1].size == 2

        # FieldArray from FieldRecord
        fra = PALEOmodel.get_array(fr)
        @test fra.values == [42.0 44.0; 43.0 45.0]
        fra_dimensions = PB.get_dimensions(fra)
        @test length(fra_dimensions) == 2
        @test fra_dimensions[1].name == "test"
        @test fra_dimensions[1].size == 2
        @test fra_dimensions[2].name == "records"
        @test fra_dimensions[2].size == 2
        
    end

    @testset "ArrayScalarData, CellSpace, UnstructuredColumnGrid" begin
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

        @test fr.records == [fill(42.0, 5, 2), fill(43.0, 5, 2)]

        @test fr[1].values == fill(42.0, 5, 2)

        # FieldArray from Field
        # fa = PALEOmodel.get_array(f, (column=1,))
        # @test length(fa.dims) == 2
        # @test fa.dims[1].name == "z"
        # @test fa.dims[2].name == "test"
        # @test size(fa.values) == (5, 2)
        # @test fa.values == fill(43.0, 5, 2)

        # fa = PALEOmodel.get_array(f, (column=1, cell=1))
        # @test length(fa.dims) == 1
        # @test fa.dims[1].name == "test"
        # @test size(fa.values) == (2, )
        # @test fa.values == fill(43.0, 2)


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

end
