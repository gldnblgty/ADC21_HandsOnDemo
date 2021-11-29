using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModelTrainer
{
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public float Prediction { get; set; }

        public float[] Score { get; set; }
    }
}
