package org.lemming.modules;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.StringTokenizer;

import org.lemming.pipeline.ExtendableTable;
import org.lemming.tools.CharBufferedReader;
import org.lemming.tools.FloatingDecimal;

import ij.IJ;
import javolution.util.FastTable;

/**
 * loading localizations into a table
 * 
 * @author Ronny Sczech
 *
 */
public class TableLoader {

	private final File file;
	private ExtendableTable table;

	public TableLoader(File file) {
		this.file = file;
		this.table = new ExtendableTable();
	}

	public void readCSV(char delimiter) {
		long start = System.currentTimeMillis();
		final Locale curLocale = Locale.getDefault();
		final Locale usLocale = new Locale("en", "US"); // setting us locale
		Locale.setDefault(usLocale);
		
		try {
			CharBufferedReader br = new CharBufferedReader(new InputStreamReader(new FileInputStream(file)));	
			
			String sCurrentLine = br.readLine();	
			if (sCurrentLine==null){ 
				br.close();
				throw new IOException("first line is null!");
			}
			while (sCurrentLine.startsWith("#")){
				sCurrentLine = br.readLine();
			}
			
			StringTokenizer s = new StringTokenizer(sCurrentLine,String.valueOf(delimiter));
			while(s.hasMoreTokens())
				table.addNewMember(s.nextToken().trim());
						
			final String[] nameArray = table.columnNames().toArray(new String[]{});
			final int lineLength = nameArray.length*19;
			
			char[] currentLine = br.readCharLine(lineLength);
			int i=0; 
			int j;
			int pos = 0;
			char[] in;
			double parsed;
			
			while (currentLine!=null){
				for (j=0; j<currentLine.length;j++){
					if (currentLine[j] == delimiter){
						in = Arrays.copyOfRange(currentLine, pos, j);
						parsed = FloatingDecimal.read(in).doubleValue();    // use own parser to circumvent use of String --> 3 times faster
						if (parsed == Math.rint(parsed))
							table.getColumn(nameArray[i++]).add((int)parsed);
						else
							table.getColumn(nameArray[i++]).add(parsed);
						pos = j+1;
					}		
				}
				in = Arrays.copyOfRange(currentLine, pos, j);
				parsed = FloatingDecimal.read(in).doubleValue();
				if (parsed == Math.rint(parsed))
					table.getColumn(nameArray[i++]).add((int)parsed);
				else
					table.getColumn(nameArray[i++]).add(parsed);
				pos = 0;
				i = 0;
				
				currentLine = br.readCharLine(lineLength);
			}
			br.close();
		} catch (IOException e) {
			IJ.error(e.getMessage());
		}
		Locale.setDefault(curLocale);
		System.out.println("Import in DataTable done in " + (System.currentTimeMillis() - start) + "ms.");
	}
	
	public ExtendableTable getTable(){
		 return table;	
	}
	
	@SuppressWarnings("unchecked")
	public void readObjects() {
		long start = System.currentTimeMillis();
		try {
			ObjectInputStream br = new ObjectInputStream(new BufferedInputStream(new FileInputStream(file)));
			int size = br.readInt();
			String[] colNames = new String[size];
			for (int k = 0; k < size; k++){
				colNames[k] = (String) br.readObject();
			}
			
			Map<String, List<Number>> readTable = new LinkedHashMap<>();
			for (int k = 0; k < size; k++)
				readTable.put(colNames[k], (FastTable<Number>) br.readObject());
			this.table = new ExtendableTable(readTable);
			br.close();
		} catch (IOException | ClassNotFoundException e) {
			IJ.error(e.getMessage());
		}
		System.out.println("Import in DataTable done in " + (System.currentTimeMillis() - start) + "ms.");
	}

}
