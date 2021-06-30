// RUN: iree-opt -split-input-file -iree-top-level-scf-to-cfg %s | IreeFileCheck %s

// CHECK-LABEL: @generic_nested_for
// While not super recommended, we do have cases of SCF constructs embedded
// in linalg.generic. This sample was reduced from a lowering of tf.pow.
// The normal -convert-scf-to-std pass will produce an illegal linalg op
// (multiple basic blocks). The -iree-top-level-scf-to-cfg should not touch it.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @generic_nested_for(%arg0: tensor<?x?x?x?xi32>, %arg1: tensor<?x?x?x?xi32>, %out0: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c6 = constant 6 : index
  %c-1_i32 = constant -1 : i32
  %c0_i32 = constant 0 : i32
  %c1_i32 = constant 1 : i32
  %c2_i32 = constant 2 : i32
  // CHECK: linalg.generic
  // CHECK: scf.for
  // CHECK: linalg.yield
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) outs(%out0 : tensor<?x?x?x?xi32>) {
  ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):  // no predecessors
    %18:3 = scf.for %arg5 = %c0 to %c6 step %c1 iter_args(%arg6 = %c1_i32, %arg7 = %arg2, %arg8 = %arg3) -> (i32, i32, i32) {
      %28 = and %arg8, %c1_i32 : i32
      %29 = cmpi eq, %28, %c1_i32 : i32
      %30 = muli %arg6, %arg7 : i32
      %31 = select %29, %30, %arg6 : i32
      %32 = muli %arg7, %arg7 : i32
      %33 = shift_right_unsigned %arg8, %c1_i32 : i32
      scf.yield %31, %32, %33 : i32, i32, i32
    }
    %19 = remi_signed %arg3, %c2_i32 : i32
    %20 = cmpi eq, %19, %c0_i32 : i32
    %21 = cmpi slt, %arg3, %c0_i32 : i32
    %22 = cmpi eq, %arg2, %c1_i32 : i32
    %23 = cmpi eq, %arg2, %c-1_i32 : i32
    %24 = select %22, %c1_i32, %c0_i32 : i32
    %25 = select %20, %c1_i32, %c-1_i32 : i32
    %26 = select %23, %25, %24 : i32
    %27 = select %21, %26, %18#0 : i32
    linalg.yield %27 : i32
  } -> tensor<?x?x?x?xi32>

  return %0 : tensor<?x?x?x?xi32>
}
