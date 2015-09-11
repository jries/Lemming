package org.lemming.modules;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.zip.ZipInputStream;

import org.lemming.pipeline.ExtendableTable;

import ij.IJ;

public class TableZipLoader implements Runnable {

	private BufferedReader br;
	private File file;
	private String delimiter;
	private ExtendableTable table;

	public TableZipLoader(File f, String d) {
		this.file = f;
		this.delimiter = d;
	}

	@Override
	public void run() {
		long start = System.currentTimeMillis();
		final Locale curLocale = Locale.getDefault();
		final Locale usLocale = new Locale("en", "US"); // setting us locale
		Locale.setDefault(usLocale);
		try {
			ZipInputStream zis=new ZipInputStream( new FileInputStream(file));
			if(zis.getNextEntry()!=null){
				br = new BufferedReader(new InputStreamReader(zis));	
				String sCurrentLine = br.readLine();	
				if (sCurrentLine==null) throw new NullPointerException("first line is null!");
				for (int i = 0; i < sCurrentLine.split(delimiter).length; i++){
					table.addNewMember("col"+i);
				}			
				
				while (sCurrentLine!=null){
					String[] s = sCurrentLine.split(delimiter);
					Map<String,Object> row = new HashMap<>(s.length);
					for (int i = 0; i < s.length; i++){
						String c = s[i].trim();
						if (c.split(".").length >1)
							row.put("col"+i, Double.parseDouble(c));
						else
							row.put("col"+i, Integer.parseInt(c));;
					}
					table.addRow(row);
					sCurrentLine = br.readLine();
				}
			}
			zis.close();
		} catch (IOException e) {
			IJ.error(e.getMessage());
		}
		Locale.setDefault(curLocale);
		System.out.println("Import in DataTable done in " + (System.currentTimeMillis() - start) + "ms.");
	}
	
	public ExtendableTable getTable(){
		 return table;	
	}

}
