static const layer_registry_entry layer_registry[] = {
{"BatchNorm", BatchNorm_final_layer_creator},                           //
{"Flatten", Flatten_final_layer_creator},                               //
{"InnerProduct", InnerProduct_final_layer_creator},                     //
{"Input", Input_final_layer_creator},                                   //
{"MemoryData", MemoryData_final_layer_creator},                         //
{"ReLU", ReLU_final_layer_creator},                                     //
{"Reshape", Reshape_final_layer_creator},                               //
{"Split", Split_final_layer_creator},                                   //
{"BinaryOp", BinaryOp_final_layer_creator},                             //
{"Padding", Padding_final_layer_creator},
{"Permute", Permute_final_layer_creator},                               //
{"Packing", Packing_final_layer_creator},
{"Cast", Cast_final_layer_creator},
{"Gemm", Gemm_final_layer_creator},                                     //
{"Convolution1D", Convolution1D_final_layer_creator},                       //
{"Pooling1D", Pooling1D_final_layer_creator},                               //
};