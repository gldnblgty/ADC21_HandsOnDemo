using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ModelTrainer
{
    public class ModelInput
    {
        [LoadColumn(0), ColumnName(@"UDI")]
        public float UDI { get; set; }

        [LoadColumn(1), ColumnName(@"Product ID")]
        public string Product_ID { get; set; }

        [LoadColumn(2), ColumnName(@"Type")]
        public string Type { get; set; }

        [LoadColumn(3), ColumnName(@"Air temperature [K]")]
        public float Air_temperature__K_ { get; set; }

        [LoadColumn(4), ColumnName(@"Process temperature [K]")]
        public float Process_temperature__K_ { get; set; }

        [LoadColumn(5), ColumnName(@"Rotational speed [rpm]")]
        public float Rotational_speed__rpm_ { get; set; }

        [LoadColumn(6), ColumnName(@"Torque [Nm]")]
        public float Torque__Nm_ { get; set; }

        [LoadColumn(7), ColumnName(@"Tool wear [min]")]
        public float Tool_wear__min_ { get; set; }

        [LoadColumn(8), ColumnName(@"Machine failure")]
        public float Machine_failure { get; set; }

        [LoadColumn(9), ColumnName(@"TWF")]
        public float TWF { get; set; }

        [LoadColumn(10), ColumnName(@"HDF")]
        public float HDF { get; set; }

        [LoadColumn(11), ColumnName(@"PWF")]
        public float PWF { get; set; }

        [LoadColumn(12), ColumnName(@"OSF")]
        public float OSF { get; set; }

        [LoadColumn(13), ColumnName(@"RNF")]
        public float RNF { get; set; }
    }
}
