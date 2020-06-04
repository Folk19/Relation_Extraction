import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

import opennlp.maxent.BasicContextGenerator;
import opennlp.maxent.ContextGenerator;
import opennlp.maxent.DataStream;
import opennlp.maxent.PlainTextByLineDataStream;
import opennlp.model.GenericModelReader;
import opennlp.model.MaxentModel;

public class Predict {
	static MaxentModel _model;
	ContextGenerator _cg = new BasicContextGenerator();

	public Predict(MaxentModel m) {
		_model = m;
	}

	/**
	 * Main method. Call as follows:
	 * <p>
	 * java Predict dataFile (modelFile)
	 * @throws UnsupportedEncodingException 
	 * @throws FileNotFoundException 
	 */

	public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
		/*這裡改訓練資料，和model檔名*/
		String dataFileName = "me_unlabel_rmu_3_new.txt", modelFileName = "me_label(1+1)PC3_7_newModel.txt";
		PrintWriter writer = new PrintWriter("me_unlabel(1+1)PC3_3_ans_new.txt", "UTF-8");
		Predict predictor = null;
		try {
			MaxentModel m = new GenericModelReader(new File(modelFileName)).getModel();
			predictor = new Predict(m);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}
		try {
	        BufferedReader br = new BufferedReader(new FileReader(dataFileName));
	        //一行一行讀
			while (br.ready()) {
				String line = br.readLine();

				String[] contexts = line.split(" ");
				double[] ocs = _model.eval(contexts); //ocs: 每一個instance的top-n結果的信心值
				
				/*輸出在這裡*/
				System.out.println("(" + _model.getBestOutcome(ocs) + ", " + ocs[ _model.getIndex(_model.getBestOutcome(ocs))] + ") ");
				writer.println("(" + _model.getBestOutcome(ocs) + ", " + ocs[ _model.getIndex(_model.getBestOutcome(ocs))] + ") ");
			}
			writer.close();
		} catch (Exception e) {
			System.out.println("Unable to read from specified file: " + modelFileName);
			System.out.println();
			e.printStackTrace();
		}
	}
}
