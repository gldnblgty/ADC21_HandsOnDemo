using System;
using System.IO;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using static ModelTrainer.ModelInput;

namespace ModelTrainer
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Let's build and save the model !!!");

            var trainData = $@"{Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName}\Data\ai4i2020.csv";

            var mlContext = new MLContext(seed: 0);
            IDataView data = mlContext.Data.LoadFromTextFile<ModelInput>(trainData, separatorChar: ',', hasHeader: true);

            //Build pipeline
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(@"Type", @"Type")
                                    .Append(mlContext.Transforms.ReplaceMissingValues(new[] 
                                    { new InputOutputColumnPair(@"UDI", @"UDI"),
                                      new InputOutputColumnPair(@"Air temperature [K]", @"Air temperature [K]"), 
                                      new InputOutputColumnPair(@"Process temperature [K]", @"Process temperature [K]"),
                                      new InputOutputColumnPair(@"Rotational speed [rpm]", @"Rotational speed [rpm]"), 
                                      new InputOutputColumnPair(@"Torque [Nm]", @"Torque [Nm]"), 
                                      new InputOutputColumnPair(@"Tool wear [min]", @"Tool wear [min]") }))
                                    // .Append(mlContext.Transforms.Text.FeaturizeText(@"Product ID", @"Product ID")) - not ONNXable using OHE instead
                                    .Append(mlContext.Transforms.Categorical.OneHotEncoding(@"Product ID", @"Product ID"))
                                    .Append(mlContext.Transforms.Concatenate(@"Features", new[] 
                                    { @"Type", @"UDI", @"Air temperature [K]", @"Process temperature [K]", @"Rotational speed [rpm]", @"Torque [Nm]", @"Tool wear [min]", @"Product ID" }))
                                    .Append(mlContext.Transforms.Conversion.MapValueToKey(@"Machine failure", @"Machine failure"))
                                    .Append(mlContext.MulticlassClassification.Trainers
                                    .OneVersusAll(binaryEstimator: mlContext.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options() 
                                    { NumberOfLeaves = 4, 
                                      MinimumExampleCountPerLeaf = 27, 
                                      NumberOfTrees = 12, 
                                      MaximumBinCountPerFeature = 410, 
                                      LearningRate = 1F, 
                                      FeatureFraction = 0.727570430993280F, 
                                      LabelColumnName = @"Machine failure", 
                                      FeatureColumnName = @"Features" }), 
                                      labelColumnName: @"Machine failure"))
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(@"PredictedLabel", @"PredictedLabel"));

            // Train the model
            ITransformer model = pipeline.Fit(data);
            Console.WriteLine($"Model training finished");

            //Save model
            var modelFile = $@"{Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName}\Model\IoTModel.zip";
            var modelonnx = $@"{Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName}\Model\IoTModel.onnx";

            //ML.net
            mlContext.Model.Save(model, data.Schema, modelFile);

            //Save model as ONNX
            using (var onnx = File.Open(modelFile, FileMode.OpenOrCreate))
            {
                mlContext.Model.ConvertToOnnx(model, data, onnx);
            }

           Console.WriteLine("The model is saved to {0}", modelFile);
        }
    }
}
