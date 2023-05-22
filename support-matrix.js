$(document).ready(function(){
    $("#training-grid").hide();
  
    $("#inference-heading-box").click(function(){
      $(this).addClass("heading-box-selected");
    //   $(this).addClass("text-white");
      $("#training-heading-box").removeClass("heading-box-selected");
    //   $("#training-heading-box").removeClass("text-white");
      $("#training-grid").hide();
      $("#inference-grid").show();
    });
  
    $("#training-heading-box").click(function(){
      $(this).addClass("heading-box-selected");
    //   $(this).addClass("text-white");
      $("#inference-heading-box").removeClass("heading-box-selected");
    //   $("#inference-heading-box").removeClass("text-white");
      $("#inference-grid").hide();
      $("#training-grid").show();
    });
  
    // Inference
    var application_type = null;
    var api = null;
    var distribution = null;
    var hardware_acceleration = null;
  
    // Training
    var platform = null;
    var framework = null;
    var framework_version = null;
    var python_version = null;
    var training_hardware = null;
  
    function updateResourceValueByClassName(class_name, value) {
      if(class_name == ".inference-application-type")
          application_type = value;
      else if(class_name == ".inference-api")
          api = value;
      else if(class_name == ".inference-distribution")
          distribution = value;
      else if(class_name == ".inference-hardware-acceleration")
          hardware_acceleration = value;
      else if(class_name == ".training-platform")
          platform = value;
      else if(class_name == ".training-framework")
          framework = value;
      else if(class_name == ".training-framework-version")
          framework_version = value;
      else if(class_name == ".training-python-version")
          python_version = value;
      else if(class_name == ".training-hardware")
          training_hardware = value;
    }
  
    function toggleButton(button, class_name) {
      if(!$(button).hasClass("gray")) {
        resource_value = null
        if($(button).hasClass("active")) {
          console.log("was active");

            $(class_name).removeClass("active");
            $(class_name).removeClass("bg-primary");
            $(class_name).removeClass("text-white");
            $(button).removeClass("active");
            $(button).removeClass("bg-primary");
            $(button).removeClass("text-white");
            console.log("After removal");
            console.log($(button).hasClass("active"));
            console.log($(button).hasClass("bg-primary"));
            console.log($(button).hasClass("text-white"));

        } else {
          console.log("wasn't active");
            $(class_name).removeClass("active");
            $(class_name).removeClass("bg-primary");
            $(class_name).removeClass("text-white");
            $(button).addClass("active");
            $(button).addClass("bg-primary");
            $(button).addClass("text-white");
            resource_value = $(button).text();
        }
        updateResourceValueByClassName(class_name, resource_value);
      }
    }
  
  //   // Inference
  
    function getInferenceInstructionText() {
      var text;
      switch (application_type) {
        case "Win32":
            switch (api) {
                case "C++":
                    switch (distribution) {
                        case "Built in Windows binaries":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://learn.microsoft.com/en-us/windows/ai/windows-ml/get-started-desktop";
                                    break;
                                case "DirectMML":
                                    text = "https://learn.microsoft.com/en-us/windows/ai/windows-ml/get-started-desktop";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/";
                                    break;
                            }
                            break;
                        case "Re-distributable package":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "DirectMML":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/";
                                    break;
                            }
                            break;
                    }
                    break;
                case "WinRT C++":
                    switch (distribution) {
                        case "Built in Windows binaries":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://learn.microsoft.com/en-us/windows/ai/windows-ml/get-started-desktop";
                                    break;
                                case "DirectML":
                                    text = "https://learn.microsoft.com/en-us/windows/ai/windows-ml/get-started-desktop";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/"
                                    break;
                            }
                            break;
                        case "Re-distributable package":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "DirectML":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/"
                                    break;
                            }
                            break;
                    }
                    break;
                case "WinRT CS":
                    switch (distribution) {
                        case "Built in Windows binaries":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/SqueezeNetObjectDetection/NETCore/cs";
                                    break;
                                case "DirectML":
                                    text = "https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/SqueezeNetObjectDetection/NETCore/cs";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/"
                                    break;
                            }
                            break;
                        case "Re-distributable package":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "DirectML":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/"
                                    break;
                            }
                            break;
                    }
                    break;
                case "WinRT Rust":
                    switch (distribution) {
                        case "Built in Windows binaries":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/RustSqueezenet";
                                    break;
                                case "DirectML":
                                    text = "https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/RustSqueezenet";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/"
                                    break;
                            }
                            break;
                        case "Re-distributable package":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "DirectML":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/"
                                    break;
                            }
                            break;
                    }
                    break;
            }
            break;
  
        case "UWP":
            switch (api) {
                case "C++":
                    switch (distribution) {
                        case "Built in Windows binaries":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://learn.microsoft.com/en-us/windows/ai/windows-ml/get-started-uwp";
                                    break;
                                case "DirectML":
                                    text = "https://learn.microsoft.com/en-us/windows/ai/windows-ml/get-started-uwp";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/";
                                    break;
                            }
                            break;
                        case "Re-distributable package":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "DirectML":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/";
                                    break;
                            }
                            break;
                    }
                    break;
                case "WinRT C++":
                    switch (distribution) {
                        case "Built in Windows binaries":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://learn.microsoft.com/en-us/windows/ai/windows-ml/get-started-uwp";
                                    break;
                                case "DirectML":
                                    text = "https://learn.microsoft.com/en-us/windows/ai/windows-ml/get-started-uwp";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/";
                                    break;
                            }
                            break;
                        case "Re-distributable package":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "DirectML":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/";
                                    break;
                            }
                            break;
                    }
                    break;
                case "WinRT CS":
                    switch (distribution) {
                        case "Built in Windows binaries":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/SqueezeNetObjectDetection/NETCore/cs";
                                    break;
                                case "DirectML":
                                    text = "https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/SqueezeNetObjectDetection/NETCore/cs";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/";
                                    break;
                            }
                            break;
                        case "Re-distributable package":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "DirectML":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/";
                                    break;
                            }
                            break;
                    }
                    break;
                case "WinRT Rust":
                    switch (distribution) {
                        case "Built in Windows binaries":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/RustSqueezenet";
                                    break;
                                case "DirectML":
                                    text = "https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/RustSqueezenet";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/";
                                    break;
                            }
                            break;
                        case "Re-distributable package":
                            switch (hardware_acceleration) {
                                case "CPU":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "DirectML":
                                    text = "https://www.nuget.org/packages/Microsoft.AI.MachineLearning";
                                    break;
                                case "Other":
                                    text = "https://onnxruntime.ai/";
                                    break;
                            }
                            break;
                    }
                    break;
            }
            break;
        
        case "Other":
            text = "https://onnxruntime.ai/";
            break;
      }
      return text;
    }
  
    function showInferenceInstruction() {
      if(application_type !== null && api !== null &&
          distribution !== null && hardware_acceleration !== null) {
              $("#inference-resource-text").hide();
              var url = getInferenceInstructionText()
              $("#inference-link").text(url);
              $("#inference-link").attr("href", url);
              $("#inference-link").show();
      } else {
        $("#inference-resource-text").show();
        $("#inference-link").hide();
      }
    }
  
    $(".inference-application-type").click(function(){
      toggleButton(this, ".inference-application-type");
    });
  
    $(".inference-api").click(function(){
      toggleButton(this, ".inference-api");
    });
  
    $(".inference-distribution").click(function(){
      toggleButton(this, ".inference-distribution");
    });
  
    $(".inference-hardware-acceleration").click(function(){
      toggleButton(this, ".inference-hardware-acceleration");
    });
  
    $(".inference-button").click(function(){
      showInferenceInstruction();
    });
  
  // Training
  
    function getTrainingInstructionText() {
      var text;
      switch (platform) {
          case "Windows":
              switch (framework) {
                  case "TensorFlow":
                      switch (framework_version) {
                          case "1.15":
                              text = "https://docs.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-windows";
                              break;
                          case "2":
                              text = "https://docs.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin";
                              break;
                              default:
                              text = "Invalid combination";
                      }
                      break;
                  case "PyTorch":
                      switch (framework_version) {
                          case "1.8":
                              text = "https://docs.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows";
                              break;
                          default:
                              text = "Invalid combination";
                      }
                      break;
              }
              break;
  
      case "WSL":
      switch (framework) {
              case "TensorFlow":
                  switch (framework_version) {
                      case "1.15":
                          text = "https://docs.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-wsl";
                          break;
                      case "2":
                          text = "https://docs.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin";
                          break;
                      default:
                          text = "Invalid commbination";
                  }
                  break;
              case "PyTorch":
              switch (framework_version) {
                  case "1.8":
                      text = "https://docs.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-wsl";
                      break;
                  default:
                      text = "Invalid combination";
              }
              break;
          }
          break;
      }
      return text;
    }
  
    function showTrainingInstruction() {
      if(platform !== null && framework !== null &&
          framework_version !== null && python_version !== null && training_hardware !== null) {
            $("#training-resource-text").hide();
            var url = getTrainingInstructionText()
            $("#training-link").text(url);
            $("#training-link").attr("href", url);
            $("#training-link").show();
      } else {
        $("#training-resource-text").show();
        $("#training-link").hide();
      }
    }
  
    $(".training-platform").click(function(){
      toggleButton(this, ".training-platform");
    });
  
    $(".training-framework").click(function(){
      $(".training-framework-version").removeClass("gray");
      toggleButton(this, ".training-framework");
      if(framework == "TensorFlow") {
          $("#fv-18").addClass("gray");
      } else if(framework == "PyTorch") {
          $("#fv-115").addClass("gray");
          $("#fv-2").addClass("gray");
      }
    });
  
    $(".training-framework-version").click(function(){
      toggleButton(this, ".training-framework-version");
    });
  
    $(".training-python-version").click(function(){
      toggleButton(this, ".training-python-version");
    });
  
    $(".training-hardware").click(function(){
      toggleButton(this, ".training-hardware");
    });
  
    $(".training-button").click(function(){
      showTrainingInstruction();
    });
  
  });