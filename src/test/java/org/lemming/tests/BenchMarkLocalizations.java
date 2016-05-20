package org.lemming.tests;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Locale;
import java.util.Map;
import java.util.StringTokenizer;

import org.lemming.pipeline.ExtendableTable;
import org.lemming.tools.CharBufferedReader;
import org.lemming.tools.FloatingDecimal;

import ij.IJ;

class BenchMarkLocalizations {
	
	private static ExtendableTable readCSV(File file, char delimiter) {
		long start = System.currentTimeMillis();
		final Locale curLocale = Locale.getDefault();
		final Locale usLocale = new Locale("en", "US"); // setting us locale
		Locale.setDefault(usLocale);
		ExtendableTable table = new ExtendableTable();
		
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
		return table;
	}
	
	
	public static void main(String[] args) {
		final ExtendableTable exp = readCSV(new File("D:\\ownCloud\\Tubulin1.csv"),'\t');
		final File file = new File("D:\\ownCloud\\Tubulin1\\frames");
		final String[] files = file.list();
		for (int num=0;num<files.length;num++){
			final String filename="D:\\ownCloud\\Tubulin1\\frames\\"+files[num];
			final ExtendableTable truth = readCSV(new File(filename),',');
			exp.addFilterExact("frame", num);
			final ExtendableTable filtered = exp.filter();
			
			for (int n=0; n<truth.getNumberOfRows();n++){
				final Map<String, Number> row = truth.getRow(n);
				final double x = row.get("xnano").doubleValue()/1000;
				final double y = row.get("ynano").doubleValue()/1000;
				double min=Double.MAX_VALUE;
				for (int m=0; m<filtered.getNumberOfRows();m++){
					final Map<String, Number> innerRow = filtered.getRow(m);
					final double tx = innerRow.get("x").doubleValue();
					final double ty = innerRow.get("y").doubleValue();
					final double dist = ((tx-x)*(tx-x))+((ty-y)*(ty-y));
					if (dist<min)min=dist;
				}
				
			}
			
			
			
			
			
			
			
			
			
			
			
			exp.resetFilter();
		}
	
		
	}

}
