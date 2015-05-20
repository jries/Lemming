package org.lemming.modules;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;
import java.util.Map;

import org.lemming.pipeline.Element;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.SingleRunModule;

public class SaveLocalizations extends SingleRunModule {

	final private Locale curLocale;
	private File file;
	private FileWriter w;
	private String inputKey;
	private long start;
	private int counter=0;

	public SaveLocalizations(File file) {
		this.curLocale = Locale.getDefault();
		this.file = file;
	}

	@Override
	public void beforeRun() {
		final Locale usLocale = new Locale("en", "US"); // setting us locale
		Locale.setDefault(usLocale);

		try {
			w = new FileWriter(file);
		} catch (IOException e) {
			e.printStackTrace();
		}

		inputKey = inputs.keySet().iterator().next();
		while (inputs.get(inputKey).isEmpty())
			pause(10);
		start = System.currentTimeMillis();
	}

	@SuppressWarnings("unchecked")
	@Override
	public void process(Map<String, Element> data) {
		Localization loc = (Localization) data.get(inputKey);
		if (loc==null) return;
		if(loc.isLast()){
			if (inputs.get(inputKey).isEmpty()){
				cancel();
				return;
			}
			inputs.get(inputKey).put(loc);
			return;
		}
		
		StringBuilder out = new StringBuilder();

		out.append(loc.getID());
		out.append(", ");
		out.append(loc.getFrame());
		out.append(", ");
		out.append(loc.getX());
		out.append(", ");
		out.append(loc.getY());
		out.append(", ");

		out.append("\n");
		try {
			w.write(out.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}		
		counter++;
	}

	@Override
	public void afterRun() {
		try {
			w.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		Locale.setDefault(curLocale);
		System.out.println("" + counter + " localizations saved in " + (System.currentTimeMillis()-start) + "ms.");
	}

}
