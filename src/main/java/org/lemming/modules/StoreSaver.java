package org.lemming.modules;

import java.beans.BeanInfo;
import java.beans.IntrospectionException;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.ElementMap;
import org.lemming.pipeline.SingleRunModule;

import ij.IJ;


public class StoreSaver extends SingleRunModule {
	
	private final File file;
	private BufferedWriter br;
	private Locale curLocale;
	private Map<String,Object> metaData;
	private static final String[] preferredOrder= new String[]{"x","y","z","sX","sY","sZ","intensity","frame"};

	public StoreSaver(File f){
		this.file = f;
	}
	
	public void putMetadata(Map<String,Object> metaData){
		this.metaData = metaData;
	}
		
	
	@Override
	public void beforeRun() {
		start = System.currentTimeMillis();
		
		curLocale = Locale.getDefault();
		final Locale usLocale = new Locale("en", "US"); // setting us locale
		Locale.setDefault(usLocale);
		Element el = inputs.get(iterator).peek();
		Set<String> headset = new LinkedHashSet<>();
		try {
			br = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file)));
			for (String key : metaData.keySet()){
			String line = "# " + key + "=" + metaData.get(key)+"\n";
				br.write(line);	
			}
			
			BeanInfo b = Introspector.getBeanInfo(el.getClass());
			for (PropertyDescriptor p : b.getPropertyDescriptors()) {
				String prop = p.getName();
				boolean test = prop.contains("class") | prop.contains("last") | prop.contains("ID");
				if (!test){
					headset.add(prop);
				}
			}
			String headline="";
			for (String aPreferredOrder : preferredOrder) {
				if (headset.contains(aPreferredOrder))
					headline += aPreferredOrder + "\t";
			}
			headline = headline.substring(0, headline.length()-1);
			br.write(headline+"\n");
		} catch (IOException | IntrospectionException e) {
			IJ.error(e.getMessage());
		}
	}

	@Override
	public Element processData(Element data) { // data not used here
		
		String converted = "";
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
		System.out.println("Saving data to "+ file.getName() + " done in " + (System.currentTimeMillis()-start) + "ms.");
	}

	@Override
	public boolean check() {
		return inputs.size()==1;
	}

}
