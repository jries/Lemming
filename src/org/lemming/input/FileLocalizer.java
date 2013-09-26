package org.lemming.input;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.data.Store;
import org.lemming.data.XYLocalization;
import org.lemming.interfaces.Localizer;
import org.lemming.outputs.NullStoreWarning;
import org.lemming.outputs.ShowMessage;

public class FileLocalizer implements Localizer {
	
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
	public void run() {
		
		if (localizations==null) {new NullStoreWarning(this.getClass().getName()); return;}
		
		String sCurrentLine;
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			long ID=0;
			while ((sCurrentLine = br.readLine()) != null) {
				String[] s = sCurrentLine.split(delimiter);
				if (s.length > 1) {
					localizations.put(new XYLocalization(Double.parseDouble(s[0]), Double.parseDouble(s[1]), ID++));					
				} else {
					new ShowMessage("The delimiter used, '" + delimiter + "', for the localization file " + filename + " can't be correct");
					break;
				}
			}
			br.close();
		} catch (FileNotFoundException e) {
			new ShowMessage(e.getMessage());
		} catch (IOException e) {
			new ShowMessage(e.getMessage());
		}
	}

	@Override
	public void setInput(Store<Frame> s) {}

	@Override
	public void setOutput(Store<Localization> s) {
		localizations = s;
	}
}
