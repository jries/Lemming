package org.lemming.modules;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.ElementMap;
import org.lemming.pipeline.SingleRunModule;

import ij.IJ;


public class StoreSaver extends SingleRunModule {
	
	private File file;
	private BufferedWriter br;
	private long start;
	private Locale curLocale;
	private Map<String,Object> metaData;

	public StoreSaver(File f){
		this.file = f;
	}
	
	public void putMetadata(Map<String,Object> metaData_){
		this.metaData = metaData_;
	}
	
	@Override
	public void beforeRun() {
		start = System.currentTimeMillis();
		
		curLocale = Locale.getDefault();
		final Locale usLocale = new Locale("en", "US"); // setting us locale
		Locale.setDefault(usLocale);
		
		try {
			br = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file)));
			for (String key : metaData.keySet()){
			String line = "# " + key + "=" + metaData.get(key)+"\n";
				br.write(line);	
			}	
		} catch (IOException e) {
			IJ.error(e.getMessage());
		}
	}

	@Override
	public Element processData(Element data) { // data not used here
		
		String converted = new String();
		if (data instanceof ElementMap){
			ElementMap me = (ElementMap) data;
			for (Entry<String, Number> entry : me.entrySet()){
				converted += entry.getValue() + ",";
			converted = converted.substring(0, converted.length()-2);
			}
		} else
			converted = data.toString();
			
		try {
			br.write(converted+"\n");
		} catch (IOException e1) {
			IJ.error(e1.getMessage());
		}
		
		if (data.isLast())
			cancel();
		
		return null;
	}

	@Override
	public void afterRun() {
		try {
			br.close();
		} catch (IOException e) {
			IJ.error(e.getMessage());
		}
		Locale.setDefault(curLocale);
		System.out.println("Reading data done in " + (System.currentTimeMillis()-start) + "ms.");
	}

	@Override
	public boolean check() {
		return inputs.size()==1;
	}

}
