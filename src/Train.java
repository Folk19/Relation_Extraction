///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2001 Chieu Hai Leong and Jason Baldridge
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//////////////////////////////////////////////////////////////////////////////   

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import opennlp.maxent.BasicEventStream;
import opennlp.maxent.GIS;
import opennlp.maxent.PlainTextByLineDataStream;
import opennlp.maxent.RealBasicEventStream;
import opennlp.maxent.io.GISModelWriter;
import opennlp.maxent.io.SuffixSensitiveGISModelWriter;
import opennlp.model.AbstractModel;
import opennlp.model.EventStream;
import opennlp.model.OnePassDataIndexer;
import opennlp.model.OnePassRealValueDataIndexer;
import opennlp.perceptron.PerceptronTrainer;

/**
 * Main class which calls the GIS procedure after building the EventStream from
 * the data.
 * 
 * @author Chieu Hai Leong and Jason Baldridge
 * @version $Revision: 1.7 $, $Date: 2008/11/06 20:00:34 $
 */
public class Train
{

	// some parameters if you want to play around with the smoothing option
	// for model training. This can improve model accuracy, though training
	// will potentially take longer and use more memory. Model size will also
	// be larger. Initial testing indicates improvements for models built on
	// small data sets and few outcomes, but performance degradation for those
	// with large data sets and lots of outcomes.

	public static boolean USE_SMOOTHING = false;
	public static double SMOOTHING_OBSERVATION = 0.1;

	/**
	 * Main method. Call as follows:
	 * <p>
	 * java CreateModel dataFile
	 * 
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException
	{

		String dataFileName = new String("me_label(7)raw_7.txt");

		String modelFileName = dataFileName.substring(0, dataFileName.lastIndexOf('.')) + "Model.txt";

		FileReader datafr = new FileReader(new File(dataFileName));
		EventStream es = new BasicEventStream(new PlainTextByLineDataStream(datafr));

		GIS.SMOOTHING_OBSERVATION = SMOOTHING_OBSERVATION;
		AbstractModel model = GIS.trainModel(8000, new OnePassRealValueDataIndexer(es, 0), USE_SMOOTHING); // 第一個參數是iterations數

		File outputFile = new File(modelFileName);
		GISModelWriter writer = new SuffixSensitiveGISModelWriter(model, outputFile);
		writer.persist();

	}

}
