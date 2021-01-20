using System;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;

namespace ImageClassifier
{
    public sealed class SqueezeNetInput
    {
        public ImageFeatureValue data_0; // shape(1,3,224,224)
    }

    public sealed class SqueezeNetOutput
    {
        public TensorFloat softmaxout_1; // shape(1,1000,1,1)
    }

    public sealed class SqueezeNetModel
    {
        private LearningModel model;
        private LearningModelSession session;
        private LearningModelBinding binding;

        public static SqueezeNetModel CreateFromFilePath(string filePath)
        {
            var learningModel = new SqueezeNetModel
            {
                model = LearningModel.LoadFromFilePath(filePath)
            };
            learningModel.session = new LearningModelSession(learningModel.model);
            learningModel.binding = new LearningModelBinding(learningModel.session);
            return learningModel;
        }

        public async Task<SqueezeNetOutput> EvaluateAsync(SqueezeNetInput input)
        {
            binding.Bind("data_0", input.data_0);
            var result = await session.EvaluateAsync(binding, "0");
            var output = new SqueezeNetOutput
            {
                softmaxout_1 = result.Outputs["softmaxout_1"] as TensorFloat
            };
            return output;
        }
    }
}
