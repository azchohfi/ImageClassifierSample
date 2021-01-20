using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;
using Windows.Graphics.Imaging;
using Windows.Media;

namespace ImageClassifier
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var rootDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            var squeezeNetModel = SqueezeNetModel.CreateFromFilePath(Path.Combine(rootDir, "squeezenet1.0-9.onnx"));

            // Load labels from JSON
            var labels = new List<string>();
            foreach (var kvp in JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(Path.Combine(rootDir, "Labels.json"))))
            {
                labels.Add(kvp.Value);
            }

            if (args.Length < 1)
                return;

            var filePath = args[0];

            // Open image file
            SqueezeNetOutput output;
            using (var fileStream = File.OpenRead(filePath))
            {
                // Convert from FileStream to ImageFeatureValue
                var decoder = await BitmapDecoder.CreateAsync(fileStream.AsRandomAccessStream());
                using var softwareBitmap = await decoder.GetSoftwareBitmapAsync();
                using var inputImage = VideoFrame.CreateWithSoftwareBitmap(softwareBitmap);
                var imageTensor = ImageFeatureValue.CreateFromVideoFrame(inputImage);

                output = await squeezeNetModel.EvaluateAsync(new SqueezeNetInput
                {
                    data_0 = imageTensor
                });
            }

            // Get result, which is a list of floats with all the probabilities for all 1000 classes of SqueezeNet
            var resultTensor = output.softmaxout_1;
            var resultVector = resultTensor.GetAsVectorView();

            // Order the 1000 results with their indexes to know which class is the highest ranked one
            List<(int index, float p)> results = new List<(int, float)>();
            for (int i = 0; i < resultVector.Count; i++)
            {
                results.Add((index: i, p: resultVector.ElementAt(i)));
            }
            results.Sort((a, b) => a.p switch
            {
                var p when p < b.p => 1,
                var p when p > b.p => -1,
                _ => 0
            });

            if (results[0].p >= 0.9f)
            {
                Console.WriteLine($"Image '{filePath}' is classified as '{labels[results[0].index]}'(p={(int)(results[0].p * 100)}%).");
            }
            else
            {
                Console.WriteLine("Sorry, but I'm not sure what this is.");
            }
        }
    }
}
