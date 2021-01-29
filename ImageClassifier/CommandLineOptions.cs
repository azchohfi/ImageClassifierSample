using CommandLine;

namespace ImageClassifier
{
    public class CommandLineOptions
    {
        [Value(index: 0, Required = true, HelpText = "Path to watch.")]
        public string Path { get; set; }

        [Option(shortName: 'e', longName: "extensions", Required = false, HelpText = "Valid image extensions.", Default = new[] { "png", "jpg", "jpeg" })]
        public string[] Extensions { get; set; }

        [Option(shortName: 'c', longName: "confidence", Required = false, HelpText = "Minimum confidence.", Default = 0.9f)]
        public float Confidence { get; set; }
    }
}
