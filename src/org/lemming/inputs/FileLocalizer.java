package org.lemming.inputs;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.data.Store;
import org.lemming.data.XYLocalization;
import org.lemming.interfaces.Localizer;
import org.lemming.outputs.ShowMessage;
import org.lemming.utils.LemMING;

public class FileLocalizer extends SO<Localization> implements Localizer {
	
	Store<Localization> localizations;
	String filename;
	String delimiter;

	/** Read localizations from <code>filename</code> and put the 
	 * values into a Store. The file must be in ASCII format, the
	 * localizations in <code>filename</code> must be in the first two
	 * columns (x then y) and there must be no header in the file.
	 * 
	 * @param filename - the full path of the file to read localizations from
	 * @param delimiter - the delimiter that is used to separated the columns in the file 
	 */
	public FileLocalizer(String filename, String delimiter) {
		this.filename = filename;
		this.delimiter = delimiter;
	}
	
	/** Read localizations from <code>filename</code> and put the 
	 * values into a Store. The file must be in ASCII format, the
	 * localizations in <code>filename</code> must be in the first two
	 * columns (x then y) and there must be no header in the file.<p>
	 * 
	 * This method will automatically determine the delimiter to use
	 * based on the file extension. If the extension is "csv" then the
	 * file is comma delimited, otherwise the file is whitespace delimited.
	 * If the file is neither comma nor whitespace delimited then use 
	 * {@link #FileLocalizer(String, String)} to specify the delimiter.
	 *  
	 * @param filename - the full path of the file to read localizations from 
	 * @see #FileLocalizer(String, String)
	 */
	public FileLocalizer(String filename) {
		this(filename, "csv".equals(filename.substring(filename.lastIndexOf('.')+1)) ? "," : "\\s");
	}

	@Override
	public boolean hasMoreOutputs() {
		return sCurrentLine != null;
	}
	
	@Override
	public Localization newOutput() {
		String[] s = sCurrentLine.split(delimiter);
		if (s.length > 1) {
			try {
				sCurrentLine = br.readLine();
			}  catch (IOException e) {
				LemMING.error(e.getMessage());
			}

			return new XYLocalization(Double.parseDouble(s[0]), Double.parseDouble(s[1]), ID++);					
		} else {
			String err = "The delimiter used, '" + delimiter + "', for the localization file " + filename + " can't be correct";
			
			LemMING.error(err);
			
			return null; // unreachable..
		}
		
	}

	BufferedReader br;
	String sCurrentLine;
	long ID;
	
	@Override
	public void run() {
		ID=0;
		try {
			br = new BufferedReader(new FileReader(filename));

			// Reads first line (needed by hasMoreOutputs)
			sCurrentLine = br.readLine();
			
			super.run();
		
			br.close();
		} catch (FileNotFoundException e) {
			new ShowMessage(e.getMessage());
		} catch (IOException e) {
			new ShowMessage(e.getMessage());
		}
	}

	@Override
	public void setInput(Store<Frame> s) {
	}
}
