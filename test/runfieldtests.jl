
using Test
using Logging

import PALEOboxes as PB
import PALEOmodel

function test_scalar_field(f::PB.Field)
    @test isnan(f.values[])
    f.values[] = 42.0

    fa = PALEOmodel.get_array(f)
    @test fa.dims == ()
    @test fa.values[] == 42.0


    fr = PALEOmodel.FieldRecord(f, Dict{Symbol, Any}(), coords_record=[])
    push!(fr, f)
    @test fr.records == [42.0]
    f.values[] = 43.0
    push!(fr, f)
    @test fr.records == [42.0, 43.0]
    @test fr[2].values == f.values

    frw = PALEOmodel.wrap_fieldrecord(
        [42.0, 43.0], 
        PB.ScalarData, 
        (),
        missing,
        PB.ScalarSpace, 
        nothing,
        Dict{Symbol, Any}();
        coords_record=[]
    )
    @test frw.records == [42.0, 43.0]
    @test frw[2].values == f.values

    fra = PALEOmodel.get_array(fr)
    @test fra.values == [42.0, 43.0]
    @test length(fra.dims) == 1
    @test fra.dims[1].name == "records"

    return nothing
end


@testset "Fields" begin

    @testset "ScalarData, ScalarSpace" begin

        f = PB.allocate_field(PB.ScalarData, (), Float64, PB.ScalarSpace, nothing; thread_safe=false, allocatenans=true)
        
        test_scalar_field(f)
    end

    @testset "ScalarData, CellSpace, no grid" begin
        # check that a CellSpace Field with no grid behaves as a ScalarSpace Field

        f = PB.allocate_field(PB.ScalarData, (), Float64, PB.CellSpace, nothing; thread_safe=false, allocatenans=true)
        
        test_scalar_field(f)
    end

    @testset "ScalarData, CellSpace, UnstructuredColumnGrid" begin
        g = PB.Grids.UnstructuredColumnGrid(ncells=5, Icolumns=[collect(1:5)])

        f = PB.allocate_field(PB.ScalarData, (), Float64, PB.CellSpace, g; thread_safe=false, allocatenans=true)
        
        f.values .= 42.0
        fr = PALEOmodel.FieldRecord(f, Dict{Symbol, Any}(), coords_record=[])
        push!(fr, f)
        f.values .= 43.0
        push!(fr, f)

        @test fr.records == [fill(42.0, 5), fill(43.0, 5)]

        @test fr[1].values == fill(42.0, 5)

        fa = PALEOmodel.get_array(fr, column=1)
        @test length(fa.dims) == 2
        @test fa.dims[1].name == "z"
        @test fa.dims[2].name == "records"
        @test size(fa.values) == (5, 2)
        @test fa.values[:, 1] == fill(42.0, 5)
        @test fa.values[:, 2] == fill(43.0, 5)
    end
  
    @testset "ArrayScalarData, ScalarSpace" begin
        f = PB.allocate_field(PB.ArrayScalarData, (PB.NamedDimension("test", 2, []), ), Float64, PB.ScalarSpace, nothing; thread_safe=false, allocatenans=true)

        @test isnan.(f.values) == [true, true]
        f.values .= [42.0, 43.0]

        fa = PALEOmodel.get_array(f)
        @test length(fa.dims) == 1
        @test fa.dims[1].name == "test"
        @test fa.dims[1].size == 2

        @test fa.values == [42.0, 43.0]


        fr = PALEOmodel.FieldRecord(f, Dict{Symbol, Any}(), coords_record=[])
        push!(fr, f)
        @test fr.records == [[42.0, 43.0]]
        f.values .= [44.0, 45.0]
        push!(fr, f)
        @test fr.records == [[42.0, 43.0], [44.0, 45.0]]
        @test fr[2].values == f.values


        # FieldArray from Field
        fa = PALEOmodel.get_array(f)
        @test fa.values == [44.0, 45.0]
        @test length(fa.dims) == 1
        @test fa.dims[1].name == "test"

        # FieldArray from FieldRecord
        fra = PALEOmodel.get_array(fr)
        @test fra.values == [42.0 44.0; 43.0 45.0]
        @test length(fra.dims) == 2
        @test fra.dims[1].name == "test"
        @test fra.dims[2].name == "records"
    end

    @testset "ArrayScalarData, CellSpace, no grid" begin
        # TODO this is possibly inconsistent with (ScalarData, CellSpace, no grid),
        # as Field.values here is a (1, 2) Array, not a (2,) Vector
        f = PB.allocate_field(PB.ArrayScalarData, (PB.NamedDimension("test", 2, []), ), Float64, PB.CellSpace, nothing; thread_safe=false, allocatenans=true)

        @test_broken size(f.values) == (2, ) # TODO should be a Vector ?
        @test size(f.values) == (1, 2)
        @test isnan.(f.values) == [true true] # TODO 1x2 Array or Vector ?
        f.values .= [42.0 43.0]  # TODO

        fr = PALEOmodel.FieldRecord(f, Dict{Symbol, Any}(), coords_record=[])
        push!(fr, f)
        @test fr.records == [[42.0 43.0]]
        f.values .= [44.0 45.0]
        push!(fr, f)
        @test fr.records == [[42.0 43.0], [44.0 45.0]]
        @test fr[2].values == f.values

        # FieldArray from Field
        fa = PALEOmodel.get_array(f)
        @test fa.values == [44.0, 45.0]
        @test length(fa.dims) == 1
        @test fa.dims[1].name == "test"
        @test fa.dims[1].size == 2

         # FieldArray from FieldRecord
         fra = PALEOmodel.get_array(fr)
         @test fra.values == [42.0 44.0; 43.0 45.0]
         @test length(fra.dims) == 2
         @test fra.dims[1].name == "test"
         @test fra.dims[2].name == "records"
        
    end

    @testset "ArrayScalarData, CellSpace, UnstructuredColumnGrid" begin
        g = PB.Grids.UnstructuredColumnGrid(ncells=5, Icolumns=[collect(1:5)])

        f = PB.allocate_field(PB.ArrayScalarData, (PB.NamedDimension("test", 2, []), ), Float64, PB.CellSpace, g; thread_safe=false, allocatenans=true)

        @test size(f.values) == (5, 2)

        f.values .= 42.0
        fr = PALEOmodel.FieldRecord(f, Dict{Symbol, Any}(), coords_record=[])
        push!(fr, f)
        f.values .= 43.0
        push!(fr, f)

        @test fr.records == [fill(42.0, 5, 2), fill(43.0, 5, 2)]

        @test fr[1].values == fill(42.0, 5, 2)

        # FieldArray from Field
        fa = PALEOmodel.get_array(f, column=1)
        @test length(fa.dims) == 2
        @test fa.dims[1].name == "z"
        @test fa.dims[2].name == "test"
        @test size(fa.values) == (5, 2)
        @test fa.values == fill(43.0, 5, 2)

        fa = PALEOmodel.get_array(f, column=1, cell=1)
        @test length(fa.dims) == 1
        @test fa.dims[1].name == "test"
        @test size(fa.values) == (2, )
        @test fa.values == fill(43.0, 2)


        # FieldArray from FieldRecord
        fra = PALEOmodel.get_array(fr, column=1)
        @test length(fra.dims) == 3
        @test fra.dims[1].name == "z"
        @test fra.dims[2].name == "test"
        @test fra.dims[3].name == "records"
        @test size(fra.values) == (5, 2, 2)
        @test fra.values[:, :, 1] == fill(42.0, 5, 2)
        @test fra.values[:, :, 2] == fill(43.0, 5, 2)

        fra = PALEOmodel.get_array(fr, column=1, cell=1)
        @test length(fra.dims) == 2
        @test fra.dims[1].name == "test"
        @test fra.dims[2].name == "records"
        @test size(fra.values) == (2, 2)
        @test fra.values[:, 1] == fill(42.0, 2)
        @test fra.values[:, 2] == fill(43.0, 2)
    end

end
