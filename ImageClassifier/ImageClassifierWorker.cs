using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;
using Windows.Graphics.Imaging;
using Windows.Media;

namespace ImageClassifier
{
    public class ImageClassifierWorker : BackgroundService
    {
        private SqueezeNetModel _squeezeNetModel;
        private readonly List<string> _labels = new List<string>();

        private readonly ILogger<ImageClassifierWorker> _logger;
        private readonly CommandLineOptions _options;

        public ImageClassifierWorker(ILogger<ImageClassifierWorker> logger, CommandLineOptions options)
        {
            _logger = logger;
            _options = options;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("Service started");

            if (!Directory.Exists(_options.Path))
            {
                _logger.LogError($"Directory \"{_options.Path}\" does not exist.");
                return;
            }

            var rootDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);

            // Load labels from JSON
            foreach (var kvp in JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(Path.Combine(rootDir, "Labels.json"))))
            {
                _labels.Add(kvp.Value);
            }

            _squeezeNetModel = SqueezeNetModel.CreateFromFilePath(Path.Combine(rootDir, "squeezenet1.0-9.onnx"));

            var count = await ProcessAllExistingFilesAsync();
            if (count > 0)
            {
                _logger.LogInformation($"Moved {count} existing files.");
            }

            _logger.LogInformation($"Listening for images created in \"{_options.Path}\"...");

            using FileSystemWatcher watcher = new FileSystemWatcher
            {
                Path = _options.Path
            };
            foreach (var extension in _options.Extensions)
            {
                watcher.Filters.Add($"*.{extension}");
            }
            watcher.Created += async (object sender, FileSystemEventArgs e) =>
            {
                await Task.Delay(1000);
                await ProcessFileAsync(e.FullPath, _options.Confidence);
            };

            watcher.EnableRaisingEvents = true;

            var tcs = new TaskCompletionSource<bool>();
            stoppingToken.Register(s => ((TaskCompletionSource<bool>)s).SetResult(true), tcs);
            await tcs.Task;

            _logger.LogInformation("Service stopped");
        }

        private async Task<int> ProcessAllExistingFilesAsync()
        {
            int count = 0;
            foreach (var extension in _options.Extensions)
            {
                foreach (var filePath in Directory.EnumerateFiles(_options.Path, $"*.{extension}", SearchOption.TopDirectoryOnly))
                {
                    if (await ProcessFileAsync(filePath, _options.Confidence))
                    {
                        count++;
                    }
                }
            }

            return count;
        }

        private async Task<bool> ProcessFileAsync(string filePath, float confidence)
        {
            try
            {
                // Open image file
                SqueezeNetOutput output;
                using (var fileStream = File.OpenRead(filePath))
                {
                    // Convert from FileStream to ImageFeatureValue
                    var decoder = await BitmapDecoder.CreateAsync(fileStream.AsRandomAccessStream());
                    using var softwareBitmap = await decoder.GetSoftwareBitmapAsync();
                    using var inputImage = VideoFrame.CreateWithSoftwareBitmap(softwareBitmap);
                    var imageTensor = ImageFeatureValue.CreateFromVideoFrame(inputImage);

                    output = await _squeezeNetModel.EvaluateAsync(new SqueezeNetInput
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

                var message = $"File: {filePath}";
                for (int i = 0; i < 3; i++)
                {
                    message += $"{Environment.NewLine}\t\"{_labels[results[i].index]}\": { results[i].p}";
                }
                _logger.LogInformation(message);

                if (results[0].p >= confidence)
                {
                    MoveFileToFolder(filePath, _labels[results[0].index]);

                    return true; // Success
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error when processing file {filePath}");
            }

            return false; // Not enough confidence or error
        }

        private void MoveFileToFolder(string filePath, string folderName)
        {
            var directory = Path.GetDirectoryName(filePath);
            var fileName = Path.GetFileName(filePath);
            var destinationDirectory = Path.Combine(directory, folderName);

            Directory.CreateDirectory(destinationDirectory);

            File.Move(filePath, Path.Combine(destinationDirectory, fileName), false);
        }
    }
}
